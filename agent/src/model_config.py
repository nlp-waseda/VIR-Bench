import os
from typing import Union, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

class ModelConfig:
    """Class to manage LLM model configuration and instantiation"""
    
    SUPPORTED_MODELS = {
        'gemini': {
            'provider': 'google',
            'models': ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash', 'gemini-2.0-flash-lite'],
            'default_model': 'gemini-2.5-flash',
            'api_key_env': 'GOOGLE_API_KEY'
        },
        'openai': {
            'provider': 'openai',
            'models': ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4o-mini'],
            'default_model': 'gpt-4o-mini',
            'api_key_env': 'OPENAI_API_KEY'
        }
    }
    
    @classmethod
    def get_supported_models(cls) -> Dict[str, Dict]:
        """Get list of supported models"""
        return cls.SUPPORTED_MODELS
    
    @classmethod
    def create_llm(cls, model_type: str = 'gemini', model_name: str = None, 
                   temperature: float = 0.7, **kwargs) -> Union[ChatGoogleGenerativeAI, ChatOpenAI]:
        """
        Create LLM instance based on specified model type

        Args:
            model_type: 'gemini' or 'openai'
            model_name: Specific model name to use (defaults if omitted)
            temperature: Parameter to control generation diversity
            **kwargs: Additional model configuration

        Returns:
            Configured LLM instance
        """
        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported: {list(cls.SUPPORTED_MODELS.keys())}")
        
        model_config = cls.SUPPORTED_MODELS[model_type]
        
        # Use default if model name is not specified
        if model_name is None:
            model_name = model_config['default_model']
        
        # Check if specified model name is in supported models list
        if model_name not in model_config['models']:
            print(f"⚠️ Warning: {model_name} is not a verified model. Using default {model_config['default_model']}.")
            model_name = model_config['default_model']
        
        # Check API key
        api_key = os.getenv(model_config['api_key_env'])
        if not api_key:
            raise ValueError(f"Required API key not set: {model_config['api_key_env']}")
        
        # Create LLM instance based on provider
        if model_config['provider'] == 'google':
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=temperature,
                **kwargs
            )
        elif model_config['provider'] == 'openai':
            return ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                temperature=temperature,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported provider: {model_config['provider']}")
    
    @classmethod
    def get_model_info(cls, model_type: str, model_name: str = None) -> Dict[str, Any]:
        """Get detailed model information"""
        if model_type not in cls.SUPPORTED_MODELS:
            return {}
        
        model_config = cls.SUPPORTED_MODELS[model_type]
        if model_name is None:
            model_name = model_config['default_model']
        
        return {
            'model_type': model_type,
            'model_name': model_name,
            'provider': model_config['provider'],
            'is_default': model_name == model_config['default_model'],
            'all_models': model_config['models'],
            'api_key_required': model_config['api_key_env']
        }
    
    @classmethod
    def validate_model_configuration(cls, model_type: str, model_name: str = None) -> Dict[str, Any]:
        """Validate model configuration"""
        validation_result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'model_info': {}
        }
        
        try:
            if model_type not in cls.SUPPORTED_MODELS:
                validation_result['errors'].append(f"Unsupported model type: {model_type}")
                return validation_result
            
            model_config = cls.SUPPORTED_MODELS[model_type]
            
            # Validate model name
            if model_name is None:
                model_name = model_config['default_model']
                validation_result['warnings'].append(f"Model name not specified, using default {model_name}")
            elif model_name not in model_config['models']:
                validation_result['warnings'].append(f"{model_name} is not a verified model")
            
            # Validate API key
            api_key = os.getenv(model_config['api_key_env'])
            if not api_key:
                validation_result['errors'].append(f"Required API key not set: {model_config['api_key_env']}")
            else:
                validation_result['warnings'].append(f"API key is set: {model_config['api_key_env']}")
            
            # Add model information
            validation_result['model_info'] = cls.get_model_info(model_type, model_name)
            
            # Valid if no errors
            validation_result['valid'] = len(validation_result['errors']) == 0
            
        except Exception as e:
            validation_result['errors'].append(f"Error occurred during validation: {str(e)}")
        
        return validation_result

def print_model_configuration_help():
    """Display model configuration help"""
    print("\n🤖 Supported LLM Models:")
    print("=" * 50)
    
    for model_type, config in ModelConfig.SUPPORTED_MODELS.items():
        print(f"\n📋 {model_type.upper()} ({config['provider']})")
        print(f"   Environment variable: {config['api_key_env']}")
        print(f"   Default model: {config['default_model']}")
        print(f"   Available models:")
        for model in config['models']:
            is_default = " (default)" if model == config['default_model'] else ""
            print(f"     - {model}{is_default}")
    
    print(f"\n💡 Usage examples:")
    print(f"   export GOOGLE_API_KEY='your-google-api-key'")
    print(f"   export OPENAI_API_KEY='your-openai-api-key'")
    print(f"   python main.py --model-type openai --model-name gpt-4o")
    print(f"   python main.py --model-type gemini --model-name gemini-pro")

if __name__ == "__main__":
    # Test script
    print_model_configuration_help()
    
    # Check environment variables
    print(f"\n🔍 Environment variable check:")
    for model_type, config in ModelConfig.SUPPORTED_MODELS.items():
        api_key = os.getenv(config['api_key_env'])
        status = "✅ Set" if api_key else "❌ Not set"
        print(f"   {config['api_key_env']}: {status}") 