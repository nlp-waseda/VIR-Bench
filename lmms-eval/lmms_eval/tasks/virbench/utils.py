import os
from pathlib import Path
import re
import random

import torch
import yaml
from decord import VideoReader, cpu
from jinja2 import Template
from PIL import Image
from difflib import SequenceMatcher


global_language = "en"


def timestamp_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, and seconds
    h, m, s = timestamp.split(":")
    # Convert hours, minutes, and total seconds (including fractions) to float and compute total seconds
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds


def load_video(video_file, duration, max_num_frames=64):
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    total_valid_frames = int(duration * fps)
    num_frames = min(max_num_frames, int(duration))

    frame_indices = [
        int(total_valid_frames / num_frames) * i for i in range(num_frames)
    ]

    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()

    return [Image.fromarray(fr).convert("RGB") for fr in frames]


def compute_frame_timestamps(duration, max_num_frames=64):
    if duration > max_num_frames:
        return [duration / max_num_frames * i for i in range(max_num_frames)]
    else:
        return [i for i in range(int(duration))]


def parse_json(pred):
    json_match = re.search(r"```json\n(.*?)\n```", pred, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            parsed_pred = yaml.safe_load(json_str)
        except yaml.YAMLError as e:
            print(f"Error parsing JSON: {e}")
            parsed_pred = None
    else:
        try:
            if "\nSummary:" in pred:
                pred = pred.split("\nSummary:")[0].strip()
            parsed_pred = yaml.safe_load(pred)
        except yaml.YAMLError as e:
            print(f"Error parsing JSON: {e}")
            parsed_pred = None
    return parsed_pred


def edge_prediction_create_input_json(doc, language):
    input = {
        "prefectures": [],
        "cities": [],
        "points_of_interest": []
    }
    unknown_poi = {
        "en": "Unknown",
        "ja": "不明"
    }

    for type, display_name, attributes in zip(doc["graph"]["nodes"]["type"], doc["graph"]["nodes"]["display_name"], doc["graph"]["nodes"]["attributes"]):
        display_name = eval(display_name)
        attributes = eval(attributes)
        if type == "ROOT":
            continue
        elif type == "PREFECTURE":
            input["prefectures"].append(display_name[language])
        elif type == "CITY":
            input["cities"].append(display_name[language])
        else:
            if display_name["ja"] == "" and display_name["en"] == "":
                continue
            if attributes["parsed_description"]["is_unknown"]:
                input["points_of_interest"].append(f"{unknown_poi[language]} ({attributes["parsed_description"]["category"]})")
            elif attributes["parsed_description"]["is_store_unknown"]:
                input["points_of_interest"].append(display_name[language].replace("(店舗不明)", "").replace("(Store Unknown)", "").strip())
            elif attributes["parsed_description"]["store_name"] != "":
                input["points_of_interest"].append(attributes["parsed_description"]["store_name"])
            else:
                input["points_of_interest"].append(display_name[language].strip())

    random.seed(42)
    random.shuffle(input["prefectures"])
    random.shuffle(input["cities"])
    random.shuffle(input["points_of_interest"])

    return input


def virbench_doc_to_text(doc, lmms_eval_specific_kwargs):
    task = lmms_eval_specific_kwargs["task"]
    language = lmms_eval_specific_kwargs["language"]
    global global_language; global_language = language
    pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    post_prompt = lmms_eval_specific_kwargs["post_prompt"]

    prompt_path = Path(
        f"{Path(__file__).resolve().parent}/prompts/{task}/{language}.j2"
    )
    prompt_template = Template(prompt_path.read_text(encoding="utf-8"))

    if task == "edge_prediction":
        input_json = edge_prediction_create_input_json(doc, language)
        prompt = prompt_template.render(input_json=input_json)
    else:
        prompt = prompt_template.render()

    return f"{pre_prompt}{prompt}\n{post_prompt}"


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)


def virbench_doc_to_target(doc):
    pass


def virbench_doc_to_visual(doc):
    with open(Path(__file__).parent / "node_prediction_debug.yaml", "r") as f:
        raw_data = f.readlines()
        safe_data = []
        for i, line in enumerate(raw_data):
            # remove function definition since yaml load cannot handle it
            if "!function" not in line:
                safe_data.append(line)
    cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
    vid_subdir_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"].get(
        "video_subdir", "videos/"
    )
    cache_dir = os.path.join(base_cache_dir, cache_name, vid_subdir_name)
    video_path = doc["video_id"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    return [video_path]


def node_prediction_process_results(doc, results):
    pred = results[0]
    
    if pred != "Gemini failed due to safety issues":
        parsed_pred = parse_json(pred)
        if not isinstance(parsed_pred, dict):
            parsed_pred = {
                "prefectures": [],
                "cities": [],
                "points_of_interest": []
            }
    else:
        parsed_pred = "Gemini failed due to safety issues"

    answer = {
        "prefectures": [],
        "cities": [],
        "points_of_interest": []
    }
    for type, display_name, attributes in zip(doc["graph"]["nodes"]["type"], doc["graph"]["nodes"]["display_name"], doc["graph"]["nodes"]["attributes"]):
        display_name = eval(display_name)
        attributes = eval(attributes)
        
        if type == "ROOT":
            continue
        elif type == "PREFECTURE":
            answer["prefectures"].append({"display_name": display_name})
        elif type == "CITY":
            answer["cities"].append({"display_name": display_name})
        else:
            if display_name["ja"] == "" and display_name["en"] == "":
                continue
            if attributes["parsed_description"]["is_unknown"]:
                continue
            answer["points_of_interest"].append({
                "duration": attributes["end"] - attributes["start"],
                "display_name": display_name,
                "category": attributes["parsed_description"]["category"],
                "types": attributes.get("details_parsed", {}).get("types", []),
                "is_store_unknown": attributes["parsed_description"]["is_store_unknown"],
                "store_name": attributes["parsed_description"]["store_name"],
            })
            
    return {
        "virbench_scores": {
            "video_id": doc["video_id"],
            "answer": answer,
            "parsed_pred": parsed_pred,     
        }
    }


def get_name_from_node_id(doc, node_id, language):
    node_id = doc["graph"]["nodes"]["id"].index(node_id)
    type = doc["graph"]["nodes"]["type"][node_id]
    display_name = eval(doc["graph"]["nodes"]["display_name"][node_id])
    attributes = eval(doc["graph"]["nodes"]["attributes"][node_id])

    unknown_poi = {
        "en": "Unknown",
        "ja": "不明"
    }
    
    if type == "ROOT":
        return "ROOT"
    elif type == "PREFECTURE":
        return display_name[language]
    elif type == "CITY":
        return display_name[language]
    else:
        if display_name["ja"] == "" and display_name["en"] == "":
            return None
        if attributes["parsed_description"]["is_unknown"]:
            return f"{unknown_poi[language]} ({attributes["parsed_description"]["category"]})"
        if attributes["parsed_description"]["is_store_unknown"]:
            return display_name[language].replace("(店舗不明)", "").replace("(Store Unknown)", "").strip()
        if attributes["parsed_description"]["store_name"] != "":
            return attributes["parsed_description"]["store_name"]
        return display_name[language].strip()


def edge_prediction_process_results(doc, results):
    pred = results[0]
    
    if pred != "Gemini failed due to safety issues":
        parsed_pred = parse_json(pred)
        if not isinstance(parsed_pred, dict):
            parsed_pred = {"edges": []}
    else:
        parsed_pred = "Gemini failed due to safety issues"

    answer = []
    for source, target, type in zip(
        doc["graph"]["edges"]["source"],
        doc["graph"]["edges"]["target"],
        doc["graph"]["edges"]["type"]
    ):
        if type == "OVERLAP":
            continue
        if source == "0": # ROOT node
            continue
        source_name, target_name = get_name_from_node_id(doc, source, global_language), get_name_from_node_id(doc, target, global_language)
        if source_name is None or target_name is None:
            continue
        if type == "INCLUSION":
            answer.append({
                "source": source_name,
                "target": target_name,
                "type": "inclusion"
            })
        elif type == "TRANSITION":
            answer.append({
                "source": source_name,
                "target": target_name,
                "type": "transition"
            })
    
    return {
        "virbench_scores": {
            "video_id": doc["video_id"],
            "answer": answer,
            "parsed_pred": parsed_pred,     
        }
    }


def parse_poi(poi, source):
    if source == "answer":
        name = []
        if poi["display_name"].get("ja", "") != "" and poi["display_name"].get("ja", "") is not None:
            name.append(poi["display_name"]["ja"].replace("(店舗不明)", "").strip().lower())
        if poi["display_name"].get("en", "") != "" and poi["display_name"].get("en", "") is not None:
            name.append(poi["display_name"]["en"].replace("(Store Unknown)", "").strip().lower())
        if poi["store_name"] != "":
            name.append(poi["store_name"].strip().lower())
        if poi["types"] != []:
            category = poi["types"]
        elif poi["category"] != "":
            category = [poi["category"]]
        else:
            category = None
    elif source == "pred":
        # Parse "Name (category)" format using regex
        match = re.match(r'^(.+?)\s*\(([^)]+)\)$', poi)
        if match:
            name = [match.group(1).strip().lower()]
            category = [match.group(2).strip()]
        else:
            name = [poi.strip().lower()]
            category = None
    else:
        raise ValueError("Invalid source for parse_poi")
    return name, category


def match_poi(names_a, categories_a, name_p, category_p, X=0.7, Y=0.55):
    for name_a in names_a:
        similarity = SequenceMatcher(None, name_a, name_p).ratio()
        if similarity > X:
            return True
        if similarity > Y and category_p in categories_a:
            return True
    return False


def evaluate_node_prediction(samples):
    results = {
        "prefectures": {
            "precision": [],
            "recall": [],
            "f1": [],
        },
        "cities": {
            "precision": [],
            "recall": [],
            "f1": [],
        },
        "points_of_interest": {
            "precision": [],
            "recall": [],
            "f1": [],
        },
        "overall": {
            "precision": [],
            "recall": [],
            "f1": [],
        }
    }

    for sample in samples:
        pred = sample["parsed_pred"]
        answer = sample["answer"]
        
        if pred == "Gemini failed due to safety issues":
            continue
        
        if pred is None:
            results["prefectures"]["precision"].append(0.0)
            results["prefectures"]["recall"].append(0.0)
            results["prefectures"]["f1"].append(0.0)
            results["cities"]["precision"].append(0.0)
            results["cities"]["recall"].append(0.0)
            results["cities"]["f1"].append(0.0)
            results["points_of_interest"]["precision"].append(0.0)
            results["points_of_interest"]["recall"].append(0.0)
            results["points_of_interest"]["f1"].append(0.0)
            results["overall"]["precision"].append(0.0)
            results["overall"]["recall"].append(0.0)
            results["overall"]["f1"].append(0.0)
            continue

        # prefectures
        tp, fp, fn = 0, 0, 0
        matched_preds = set()
        for pref_a in answer["prefectures"]:
            match_found = False
            for idx, pref_p in enumerate(pred.get("prefectures", [])):
                if idx in matched_preds:
                    continue
                if pref_p in [pref_a["display_name"]["ja"], pref_a["display_name"]["en"]]:
                    tp += 1
                    match_found = True
                    matched_preds.add(idx)
                    break
            if not match_found:
                fn += 1
        fp = len(pred.get("prefectures", [])) - len(matched_preds)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        results["prefectures"]["precision"].append(precision)
        results["prefectures"]["recall"].append(recall)
        results["prefectures"]["f1"].append(f1)

        # cities
        tp, fp, fn = 0, 0, 0
        matched_preds = set()
        for pref_a in answer["cities"]:
            match_found = False
            for idx, pref_p in enumerate(pred.get("cities", [])):
                if idx in matched_preds:
                    continue
                if pref_p in [pref_a["display_name"]["ja"], pref_a["display_name"]["en"]]:
                    tp += 1
                    match_found = True
                    matched_preds.add(idx)
                    break
            if not match_found:
                fn += 1
        fp = len(pred.get("cities", [])) - len(matched_preds)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        results["cities"]["precision"].append(precision)
        results["cities"]["recall"].append(recall)
        results["cities"]["f1"].append(f1)

        # POI
        tp, fp, fn = 0, 0, 0
        matched_preds = set()
        for pref_a in answer["points_of_interest"]:
            names_a, categories_a = parse_poi(pref_a, "answer")
            match_found = False
            for idx, pref_p in enumerate(pred.get("points_of_interest", [])):
                if idx in matched_preds:
                    continue
                name_p, category_p = parse_poi(pref_p, "pred")
                if match_poi(names_a, categories_a, name_p[0], category_p[0] if category_p else None):
                    tp += 1
                    match_found = True
                    matched_preds.add(idx)
                    break
            if not match_found:
                fn += 1
        fp = len(pred.get("points_of_interest", [])) - len(matched_preds)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        results["points_of_interest"]["precision"].append(precision)
        results["points_of_interest"]["recall"].append(recall)
        results["points_of_interest"]["f1"].append(f1)
        
        # calculate weighted average for overall metrics
        num_pref = len(answer["prefectures"])
        num_cities = len(answer["cities"])
        num_pois = len(answer["points_of_interest"])
        results["overall"]["precision"].append(
            (num_pref * results["prefectures"]["precision"][-1] +
             num_cities * results["cities"]["precision"][-1] +
             num_pois * results["points_of_interest"]["precision"][-1]) /
            (num_pref + num_cities + num_pois)
        )
        results["overall"]["recall"].append(
            (num_pref * results["prefectures"]["recall"][-1] +
             num_cities * results["cities"]["recall"][-1] +
             num_pois * results["points_of_interest"]["recall"][-1]) /
            (num_pref + num_cities + num_pois)
        )
        results["overall"]["f1"].append(
            (num_pref * results["prefectures"]["f1"][-1] +
             num_cities * results["cities"]["f1"][-1] +
             num_pois * results["points_of_interest"]["f1"][-1]) /
            (num_pref + num_cities + num_pois)
        )

    # aggregate results
    for category in results:
        for metric in results[category]:
            values = results[category][metric]
            results[category][metric] = sum(values) / len(values) if values else 0.0

    return results


def node_prediction_aggregate_results(results):
    evaluation_results = evaluate_node_prediction(results)
    return evaluation_results


def evaluate_edge_prediction(samples):
    results = {
        "inclusion": {
            "precision": [],
            "recall": [],
            "f1": [],
        },
        "transition": {
            "precision": [],
            "recall": [],
            "f1": [],
        },
        "overall": {
            "precision": [],
            "recall": [],
            "f1": [],
        }
    }

    for sample in samples:
        pred = sample["parsed_pred"]["edges"] if sample["parsed_pred"] and "edges" in sample["parsed_pred"] else None
        answer = sample["answer"]

        if pred == "Gemini failed due to safety issues":
            continue

        if pred is None:
            for category in results:
                results[category]["precision"].append(0.0)
                results[category]["recall"].append(0.0)
                results[category]["f1"].append(0.0)
            continue

        inclusion_edges_answer = [edge for edge in answer if edge["type"] == "inclusion"]
        transition_edges_answer = [edge for edge in answer if edge["type"] == "transition"]
        inclusion_edges_pred = [edge for edge in pred if isinstance(edge, dict) and edge.get("type") == "inclusion"]
        transition_edges_pred = [edge for edge in pred if isinstance(edge, dict) and edge.get("type") == "transition"]
        
        # Calculate inclusion metrics
        inclusion_tp = sum(1 for edge in inclusion_edges_answer if edge in inclusion_edges_pred)
        inclusion_fp = len(inclusion_edges_pred) - inclusion_tp
        inclusion_fn = len(inclusion_edges_answer) - inclusion_tp
        
        inclusion_precision = inclusion_tp / (inclusion_tp + inclusion_fp) if (inclusion_tp + inclusion_fp) > 0 else 0.0
        inclusion_recall = inclusion_tp / (inclusion_tp + inclusion_fn) if (inclusion_tp + inclusion_fn) > 0 else 0.0
        inclusion_f1 = (2 * inclusion_precision * inclusion_recall / (inclusion_precision + inclusion_recall)) if (inclusion_precision + inclusion_recall) > 0 else 0.0
        
        results["inclusion"]["precision"].append(inclusion_precision)
        results["inclusion"]["recall"].append(inclusion_recall)
        results["inclusion"]["f1"].append(inclusion_f1)
        
        # Calculate transition metrics
        transition_tp = sum(1 for edge in transition_edges_answer if edge in transition_edges_pred)
        transition_fp = len(transition_edges_pred) - transition_tp
        transition_fn = len(transition_edges_answer) - transition_tp
        
        transition_precision = transition_tp / (transition_tp + transition_fp) if (transition_tp + transition_fp) > 0 else 0.0
        transition_recall = transition_tp / (transition_tp + transition_fn) if (transition_tp + transition_fn) > 0 else 0.0
        transition_f1 = (2 * transition_precision * transition_recall / (transition_precision + transition_recall)) if (transition_precision + transition_recall) > 0 else 0.0
        
        results["transition"]["precision"].append(transition_precision)
        results["transition"]["recall"].append(transition_recall)
        results["transition"]["f1"].append(transition_f1)
        
        # Calculate overall metrics
        overall_tp = inclusion_tp + transition_tp
        overall_fp = inclusion_fp + transition_fp
        overall_fn = inclusion_fn + transition_fn
        
        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
        overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)) if (overall_precision + overall_recall) > 0 else 0.0
        
        results["overall"]["precision"].append(overall_precision)
        results["overall"]["recall"].append(overall_recall)
        results["overall"]["f1"].append(overall_f1)

    # aggregate results
    for category in results:
        for metric in results[category]:
            values = results[category][metric]
            results[category][metric] = sum(values) / len(values) if values else 0.0

    return results


def edge_prediction_aggregate_results(results):
    evaluation_results = evaluate_edge_prediction(results)
    return evaluation_results