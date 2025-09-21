import io
import json
import os
import pathlib
import re
import time
from typing import List, Tuple

from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

try:
    from google import genai
    from google.genai import types

    NUM_SECONDS_TO_SLEEP = 60
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=GOOGLE_API_KEY)

except Exception as e:
    eval_logger.error(f"Error importing generativeai: {str(e)}")
    client = None

try:
    import soundfile as sf
except Exception as e:
    eval_logger.warning(
        f"Error importing soundfile, audio generation will not work: {str(e)}"
    )


@register_model("gemini_api")
class GeminiAPI(lmms):
    def __init__(
        self,
        model_version: str = "gemini-2.5-flash",
        # modality: str = "image",
        timeout: int = 120,
        continual_mode: bool = True,
        response_persistent_folder: str = "./logs/gemini_persistent_folder",
        interleave: bool = False,
        thinking_budget: int = 0,
        off_audio: bool = False,
        # We will cache the Gemini API response in this path and use it for future requests
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.timeout = timeout
        self.continual_mode = continual_mode
        self.response_persistent_file = ""
        self.interleave = interleave
        # if self.continual_mode and response_persistent_folder is None:
        #     raise ValueError("Continual mode requires a persistent path for the response. We will cache the Gemini API response in this path and use it for future requests. Please provide a valid path.")
        if self.continual_mode:
            self.response_persistent_folder = response_persistent_folder
            if not os.path.exists(self.response_persistent_folder):
                os.makedirs(self.response_persistent_folder)
            self.response_persistent_file = os.path.join(
                self.response_persistent_folder, f"{self.model_version}_response.json"
            )
        self.thinking_budget = thinking_budget

        if os.path.exists(self.response_persistent_file):
            with open(self.response_persistent_file, "r") as f:
                self.response_cache = json.load(f)
            self.cache_mode = "resume"
        else:
            self.response_cache = {}
            self.cache_mode = "start"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert self.continual_mode is False, (
                "Continual mode is not supported with distributed inference."
            )
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(
                    f"Using {accelerator.num_processes} devices with data parallelism"
                )
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device
        
        self.off_audio = off_audio

        # self.modality = modality

    def free_video(self):
        for f in client.files.list():
            client.files.delete(name=f.name)

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def get_image_size(self, image):
        # Create a BytesIO object to store the image bytes
        img_byte_array = io.BytesIO()

        # Save the image to the BytesIO object
        image.save(img_byte_array, format="PNG")

        # Get the size of the BytesIO object
        img_size = img_byte_array.tell()

        return img_size

    def encode_video(self, video_path):
        uploaded_obj = client.files.upload(file=video_path)
        file_state = client.files.list()[0].state
        while file_state != types.FileState.ACTIVE:
            if file_state == types.FileState.FAILED:
                eval_logger.error(f"Video upload failed for {video_path}")
                raise ValueError(f"Video upload failed for {video_path}")
            eval_logger.info(
                f"Video upload in progress for {video_path}, current state: {file_state.name}"
            )
            time.sleep(NUM_SECONDS_TO_SLEEP)
            file_state = client.files.list()[0].state
        return uploaded_obj

    def encode_audio(self, audio):
        audio_io = io.BytesIO()
        sf.write(audio_io, audio["array"], audio["sampling_rate"], format="WAV")
        return client.files.upload(audio_io, mime_type="audio/wav")

    def convert_modality(self, images):
        for idx, img in enumerate(images):
            if isinstance(img, dict) and "sampling_rate" in img:  # audio
                audio = self.encode_audio(img)
                images[idx] = audio
            elif isinstance(img, str):  # video
                try:
                    images[idx] = self.encode_video(img)
                except Exception as e:
                    eval_logger.error(f"Error converting video: {str(e)}")
        return images

    def construct_interleaved_input(self, content, media):
        pattern = r"<media_(\d+)>"
        parts = re.split(pattern, content)
        result = []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                if part == "":
                    continue
                result.append(part)
            else:
                result.append(media[int(part)])

        return result

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="Model Responding"
        )

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [
            reg.args for reg in requests
        ]:
            if self.continual_mode and self.cache_mode == "resume":
                doc_uuid = get_uuid(task, split, doc_id)
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content and content != "":
                        res.append(content)
                        pbar.update(1)
                        continue

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            
            if self.model_version == "gemini-2.5-pro":
                gen_kwargs["max_new_tokens"] = self.thinking_budget

            config = types.GenerateContentConfig(
                max_output_tokens=gen_kwargs["max_new_tokens"],
                thinking_config=types.ThinkingConfig(
                    thinking_budget=self.thinking_budget
                ),
                temperature=gen_kwargs["temperature"],
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_CIVIC_INTEGRITY",
                        threshold="BLOCK_NONE",
                    )
                ],
            )

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            if self.off_audio:
                new_visuals = []
                for visual in visuals:
                    audio_removed = visual.replace(".mp4", ".no_audio.mp4")
                    import os, subprocess
                    if not os.path.exists(audio_removed):
                        cmd = [
                            "ffmpeg",
                            "-i", visual,
                            "-c:v", "copy",
                            "-an",
                            audio_removed,
                        ]
                        subprocess.run(cmd, check=True)
                    new_visuals.append(audio_removed)
                visuals = new_visuals
            visuals = self.convert_modality(visuals)
            use_fixed_video = False

            if self.interleave:
                message = self.construct_interleaved_input(contexts, visuals)
            else:
                message = [contexts] + visuals

            for attempt in range(5):
                try:
                    content = client.models.generate_content(
                        model=self.model_version,
                        contents=message,
                        config=config,
                    )
                    content = content.text
                    assert content is not None
                    break
                except Exception as e:
                    eval_logger.info(
                        f"Attempt {attempt + 1} failed with error: {str(e)}"
                    )
                    # fix the video
                    if str(e) == "" and not use_fixed_video:
                        visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
                        visuals = self.flatten(visuals)
                        new_visuals = []
                        for visual in visuals:
                            if isinstance(visual, str):
                                if self.off_audio:
                                    fixed = visual.replace(".mp4", ".fixed_gemini.no_audio.mp4")
                                else:
                                    fixed = visual.replace(".mp4", ".fixed_gemini.mp4")
                                import os, subprocess
                                if not os.path.exists(fixed):
                                    if self.off_audio:
                                        cmd = [
                                            "ffmpeg",
                                            "-i", visual,
                                            "-vf", "scale=-2:240",
                                            "-c:v", "libopenh264",
                                            "-preset", "veryfast",
                                            "-b:v", "400k",
                                            "-pix_fmt", "yuv420p",
                                            "-an",
                                            fixed,
                                        ]
                                    else:
                                        cmd = [
                                            "ffmpeg",
                                            "-i", visual,
                                            "-vf", "scale=-2:240",
                                            "-c:v", "libopenh264",
                                            "-preset", "veryfast",
                                            "-b:v", "400k",
                                            "-pix_fmt", "yuv420p",
                                            "-c:a", "copy",
                                            fixed,
                                        ]
                                    subprocess.run(cmd, check=True)
                                new_visuals.append(fixed)
                        visuals = self.convert_modality(new_visuals)
                        message = [contexts] + visuals
                        use_fixed_video = True
                    if isinstance(e, ValueError):
                        try:
                            eval_logger.info(
                                f"Prompt feed_back: {content.prompt_feedback}"
                            )
                            content = ""
                            break
                        except Exception:
                            pass
                    if (
                        attempt < 5 - 1
                    ):  # If we have retries left, sleep and then continue to next attempt
                        time.sleep(NUM_SECONDS_TO_SLEEP)
                    else:  # If this was the last attempt, log and return empty
                        eval_logger.error(
                            f"All 5 attempts failed. Last error message: {str(e)}"
                        )
                        content = "Gemini failed due to safety issues"
            res.append(content)
            pbar.update(1)

            self.free_video()

            if self.continual_mode is True:  # Cache the response
                doc_uuid = get_uuid(task, split, doc_id)
                self.response_cache[doc_uuid] = content
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError(
            "TODO: Implement multi-round generation for Gemini API"
        )

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "Gemini API not support"

    def get_image_audio_text_interleaved_messsage(
        self, image_path, audio_path, question
    ):
        # image_path for list of image path
        # audio_path for list of audio path
        # question for question

        # fixed image token and no audio in text
        for index in range(1, 1 + len(image_path)):
            question = question.replace(f"[img{index}]", "<image>")
        for index in range(1, 1 + len(audio_path)):
            question = question.replace(f"[audio{index}]", "<audio>")

        text = question

        info_list = []
        image_counter = 0
        audio_counter = 0
        for part in re.split(r"(<image>|<audio>)", text):
            if part == "<image>":
                info_list.append(Image.open(image_path[image_counter]))
                image_counter += 1
            elif part == "<audio>":
                info_list.append(
                    {
                        "mime_type": "audio/wav",
                        "data": pathlib.Path(audio_path[audio_counter]).read_bytes(),
                    }
                )
                audio_counter += 1
            else:
                if part == " ":
                    continue
                info_list.append(part)

        return info_list

    def get_video_audio_text_interleaved_message(
        self, video_path, audio_path, question
    ):
        # image_path for list of image path
        # audio_path for list of audio path
        # question for question

        # fixed video token and no audio in text
        for index in range(1, 1 + len(video_path)):
            question = question.replace(f"[video{index}]", "<video>")
        for index in range(1, 1 + len(audio_path)):
            question = question.replace(f"[audio{index}]", "<audio>")

        text = question

        info_list = []
        video_counter = 0
        audio_counter = 0
        for part in re.split(r"(<video>|<audio>)", text):
            if part == "<video>":
                current_video_file_name = video_path[video_counter]
                current_video_file = client.files.upload(file=current_video_file_name)
                while current_video_file.state.name == "processing":
                    print("uploading file")
                    time.sleep(5)
                    current_video_file = client.files.get(name=current_video_file.name)
                if current_video_file.state.name == "FAILED":
                    print("uploading file failed, next question")
                    return 0
                info_list.append(current_video_file)
                video_counter += 1
            elif part == "<audio>":
                info_list.append(
                    {
                        "mime_type": "audio/wav",
                        "data": pathlib.Path(audio_path[audio_counter]).read_bytes(),
                    }
                )
                audio_counter += 1
            else:
                if part == " ":
                    continue
                info_list.append(part)

        return info_list
