import tiktoken
from typing import Dict, List, Any
from datetime import datetime
import json

class TokenCounter:
    """Class to count LLM token usage and calculate costs"""
    
    # Model pricing per 1K tokens (USD)
    PRICING = {
        # OpenAI models
        'gpt-4o': {'input': 0.0025, 'output': 0.01},
        'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
        'gpt-4': {'input': 0.03, 'output': 0.06},
        'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
        'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
        'gpt-3.5-turbo-1106': {'input': 0.001, 'output': 0.002},
        
        'models/gemini-2.5-flash-preview-04-17': {'input': 0.0003, 'output': 0.0025},
        'models/gemini-2.5-pro-preview-04-17': {'input': 0.00125, 'output': 0.01},
        
        # Default for unknown models
        'default': {'input': 0.001, 'output': 0.002}
    }
    
    def __init__(self):
        self.usage_by_agent = {}
        self.total_usage = {
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'total_tokens': 0,
            'total_cost_usd': 0.0
        }
        self.usage_log = []
        
        # Initialize tiktoken for OpenAI models
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = None
    
    def count_tokens(self, text: str, model: str = None) -> int:
        """テキストのトークン数をカウント"""
        if not text:
            return 0
            
        # For OpenAI models, use tiktoken
        if self.encoding and (not model or 'gpt' in model.lower()):
            try:
                return len(self.encoding.encode(text))
            except:
                pass
        
        # Fallback: estimate based on character count
        # Rough estimation: ~4 characters per token for English, ~2 for Japanese
        char_count = len(text)
        japanese_chars = sum(1 for c in text if '\u3000' <= c <= '\u9fff' or '\uff00' <= c <= '\uffef')
        english_chars = char_count - japanese_chars
        
        estimated_tokens = (japanese_chars / 2) + (english_chars / 4)
        return int(estimated_tokens)
    
    def track_usage(self, agent_name: str, model_name: str, input_text: str, output_text: str, 
                    input_tokens: int = None, output_tokens: int = None):
        """エージェントのトークン使用量を記録
        
        Args:
            agent_name: エージェント名
            model_name: モデル名
            input_text: 入力テキスト（トークンカウント用）
            output_text: 出力テキスト（トークンカウント用）
            input_tokens: 既知の入力トークン数（Geminiのusage_metadataから）
            output_tokens: 既知の出力トークン数（Geminiのusage_metadataから）
        """
        # If token counts are provided (e.g., from Gemini's usage_metadata), use them
        if input_tokens is None:
            input_tokens = self.count_tokens(input_text, model_name)
        if output_tokens is None:
            output_tokens = self.count_tokens(output_text, model_name)
        
        total_tokens = input_tokens + output_tokens
        
        # Calculate cost
        cost = self._calculate_cost(model_name, input_tokens, output_tokens)
        
        # Create usage record
        usage_record = {
            'timestamp': datetime.now().isoformat(),
            'agent_name': agent_name,
            'model_name': model_name,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'cost_usd': cost
        }
        
        # Update agent-specific usage
        if agent_name not in self.usage_by_agent:
            self.usage_by_agent[agent_name] = {
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_tokens': 0,
                'total_cost_usd': 0.0,
                'invocations': 0,
                'model_used': model_name
            }
        
        agent_usage = self.usage_by_agent[agent_name]
        agent_usage['total_input_tokens'] += input_tokens
        agent_usage['total_output_tokens'] += output_tokens
        agent_usage['total_tokens'] += total_tokens
        agent_usage['total_cost_usd'] += cost
        agent_usage['invocations'] += 1
        
        # Update total usage
        self.total_usage['total_input_tokens'] += input_tokens
        self.total_usage['total_output_tokens'] += output_tokens
        self.total_usage['total_tokens'] += total_tokens
        self.total_usage['total_cost_usd'] += cost
        
        # Add to log
        self.usage_log.append(usage_record)
        
        # Print real-time usage
        print(f"📊 Token usage - {agent_name}: {input_tokens} in / {output_tokens} out (${cost:.4f})")
    
    def _calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost from token count"""
        # Find pricing for the model
        pricing = self.PRICING.get('default')
        
        for model_key in self.PRICING:
            if model_key in model_name.lower():
                pricing = self.PRICING[model_key]
                break
        
        # Calculate cost (pricing is per 1K tokens)
        input_cost = (input_tokens / 1000) * pricing['input']
        output_cost = (output_tokens / 1000) * pricing['output']
        
        return input_cost + output_cost
    
    def get_summary(self) -> Dict[str, Any]:
        """使用量サマリーを取得"""
        return {
            'token_usage_by_agent': self.usage_by_agent,
            'total_usage': self.total_usage,
            'usage_details': self.usage_log,
            'summary_stats': {
                'total_agents_used': len(self.usage_by_agent),
                'total_llm_calls': len(self.usage_log),
                'average_tokens_per_call': self.total_usage['total_tokens'] / len(self.usage_log) if self.usage_log else 0,
                'cost_breakdown': self._get_cost_breakdown()
            }
        }
    
    def _get_cost_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by agent"""
        breakdown = {}
        for agent_name, usage in self.usage_by_agent.items():
            breakdown[agent_name] = usage['total_cost_usd']
        return breakdown
    
    def save_to_file(self, filepath: str):
        """トークン使用量をJSONファイルに保存"""
        summary = self.get_summary()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def print_summary(self):
        """使用量サマリーを表示"""
        print("\n" + "="*60)
        print("📊 TOKEN USAGE SUMMARY")
        print("="*60)
        
        # Agent breakdown
        print("\n🤖 Usage by Agent:")
        for agent_name, usage in self.usage_by_agent.items():
            print(f"\n  {agent_name}:")
            print(f"    - Invocations: {usage['invocations']}")
            print(f"    - Input tokens: {usage['total_input_tokens']:,}")
            print(f"    - Output tokens: {usage['total_output_tokens']:,}")
            print(f"    - Total tokens: {usage['total_tokens']:,}")
            print(f"    - Cost: ${usage['total_cost_usd']:.4f}")
        
        # Total summary
        print(f"\n💰 TOTAL USAGE:")
        print(f"  - Total input tokens: {self.total_usage['total_input_tokens']:,}")
        print(f"  - Total output tokens: {self.total_usage['total_output_tokens']:,}")
        print(f"  - Total tokens: {self.total_usage['total_tokens']:,}")
        print(f"  - Total cost: ${self.total_usage['total_cost_usd']:.4f}")
        print("="*60 + "\n")

# Global token counter instance
token_counter = TokenCounter()