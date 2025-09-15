import os
import json
import datetime
from typing import Dict, List, Any
from pathlib import Path

class OutputManager:
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or f"experiment_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path("results") / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "raw_data").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "tool_logs").mkdir(exist_ok=True)  # ツールログ用ディレクトリ
        
    def save_experiment_config(self, config: Dict[str, Any]):
        """Save experiment configuration"""
        config_path = self.output_dir / "experiment_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def save_model_config(self, model_info: Dict[str, Any]):
        """Save model configuration"""
        model_config_path = self.output_dir / "model_config.json"
        with open(model_config_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    def save_agent_result(self, agent_name: str, result: Dict[str, Any]):
        """Save individual agent result"""
        timestamp = datetime.datetime.now().isoformat()
        result_with_timestamp = {
            "timestamp": timestamp,
            "agent_name": agent_name,
            **result
        }
        
        filename = f"{timestamp.replace(':', '-')}_{agent_name.lower().replace(' ', '_')}.json"
        filepath = self.output_dir / "raw_data" / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_with_timestamp, f, indent=2, ensure_ascii=False)
    
    def save_tool_logs(self, tool_logs: List[Dict[str, Any]], agent_name: str = None):
        """Save tool execution logs"""
        timestamp = datetime.datetime.now().isoformat()
        
        # ツールログファイル名
        if agent_name:
            filename = f"{timestamp.replace(':', '-')}_tool_logs_{agent_name.lower().replace(' ', '_')}.json"
        else:
            filename = f"{timestamp.replace(':', '-')}_tool_logs.json"
        
        filepath = self.output_dir / "tool_logs" / filename
        
        # 詳細なツールログデータを構築
        tool_log_data = {
            "timestamp": timestamp,
            "agent_name": agent_name,
            "total_tool_executions": len(tool_logs),
            "successful_executions": len([log for log in tool_logs if log.get('success', False)]),
            "failed_executions": len([log for log in tool_logs if not log.get('success', True)]),
            "total_execution_time": sum(log.get('execution_time_seconds', 0) for log in tool_logs),
            "tool_executions": tool_logs
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tool_log_data, f, indent=2, ensure_ascii=False)
        
        # コンソール出力
        print(f"📊 ツールログ保存: {len(tool_logs)} 件の実行ログ -> {filename}")
        if tool_logs:
            successful = tool_log_data["successful_executions"]
            failed = tool_log_data["failed_executions"]
            total_time = tool_log_data["total_execution_time"]
            print(f"   成功: {successful}, 失敗: {failed}, 総実行時間: {total_time:.2f}秒")
    
    def save_orchestration_log(self, orchestration_decisions: List[Dict[str, Any]]):
        """Save orchestration decision log"""
        log_path = self.output_dir / "orchestration_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(orchestration_decisions, f, indent=2, ensure_ascii=False)
    
    def save_final_plan(self, plan_data: Dict[str, Any]):
        """Save final travel plan"""
        # Save as JSON
        json_path = self.output_dir / "final_travel_plan.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(plan_data, f, indent=2, ensure_ascii=False)
        
        # Generate two separate reports
        # 1. Travel guide for travelers
        travel_guide_path = self.output_dir / "reports" / "travel_guide.md"
        self._generate_travel_guide(plan_data, travel_guide_path)
        
        # 2. Technical execution log
        execution_log_path = self.output_dir / "reports" / "execution_log.md"
        self._generate_execution_log(plan_data, execution_log_path)
    
    def _generate_travel_guide(self, plan_data: Dict[str, Any], filepath: Path):
        """Generate traveler-friendly guide without technical details"""
        report_content = f"""# Travel Guide

**Destination**: {plan_data.get('location_context', 'N/A')}
**Travel Dates**: {plan_data.get('experiment_constraints', {}).get('days', 'N/A')} days
**Group Size**: {plan_data.get('experiment_constraints', {}).get('people_count', 'N/A')} people
**Budget**: ${plan_data.get('experiment_constraints', {}).get('budget', 'N/A')} USD

---

{plan_data.get('comprehensive_summary', 'No travel plan generated')}

---

## Budget Breakdown
{self._format_budget_analysis(plan_data.get('budget_analysis', {}))}

---

*Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    def _generate_execution_log(self, plan_data: Dict[str, Any], filepath: Path):
        """Generate technical execution log with all metadata"""
        # ツールログサマリーを追加
        tool_logs_summary = self._generate_tool_logs_summary()
        
        # トークン使用量サマリーを追加
        token_usage_summary = self._generate_token_usage_summary()
        
        report_content = f"""# Technical Execution Log

**Experiment**: {self.experiment_name}
**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
- **Model Type**: {plan_data.get('model_info', {}).get('model_type', 'N/A')}
- **Model Name**: {plan_data.get('model_info', {}).get('model_name', 'N/A')}
- **Group Size**: {plan_data.get('experiment_constraints', {}).get('people_count', 'N/A')} people
- **Duration**: {plan_data.get('experiment_constraints', {}).get('days', 'N/A')} days
- **Budget**: ${plan_data.get('experiment_constraints', {}).get('budget', 'N/A')} USD
- **Location**: {plan_data.get('location_context', 'N/A')}

## Token Usage and Cost Summary
{token_usage_summary}

## Tool Execution Summary
{tool_logs_summary}

## Execution Summary
- **Total Agents Executed**: {plan_data.get('execution_summary', {}).get('total_agents_executed', 'N/A')}
- **Successful Executions**: {plan_data.get('execution_summary', {}).get('successful_executions', 'N/A')}
- **POIs Requested**: {plan_data.get('execution_summary', {}).get('total_pois_requested', 'N/A')}
- **POIs Found**: {plan_data.get('execution_summary', {}).get('total_pois_found', 'N/A')}
- **POIs Selected**: {plan_data.get('execution_summary', {}).get('final_selected_pois', 'N/A')}

## Orchestration Decisions
{self._format_orchestration_decisions(plan_data.get('orchestration_decisions', []))}

## Agent Execution Details
{self._format_agent_execution_details(plan_data)}

## File Locations
- **Experiment Config**: `{self.output_dir}/experiment_config.json`
- **Model Config**: `{self.output_dir}/model_config.json`
- **Final Plan JSON**: `{self.output_dir}/final_travel_plan.json`
- **Tool Logs**: `{self.output_dir}/tool_logs/`
- **Raw Agent Data**: `{self.output_dir}/raw_data/`
- **Orchestration Log**: `{self.output_dir}/orchestration_log.json`
- **Token Usage Summary**: `{self.output_dir}/token_usage_summary.json`
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    def _generate_tool_logs_summary(self) -> str:
        """Generate summary of tool execution logs"""
        tool_logs_dir = self.output_dir / "tool_logs"
        
        if not tool_logs_dir.exists():
            return "No tool execution logs found."
        
        # すべてのツールログファイルを読み込んで統計を生成
        total_executions = 0
        successful_executions = 0
        failed_executions = 0
        total_time = 0.0
        tool_usage_count = {}
        
        for log_file in tool_logs_dir.glob("*.json"):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                    
                total_executions += log_data.get('total_tool_executions', 0)
                successful_executions += log_data.get('successful_executions', 0)
                failed_executions += log_data.get('failed_executions', 0)
                total_time += log_data.get('total_execution_time', 0)
                
                # 各ツールの使用回数をカウント
                for execution in log_data.get('tool_executions', []):
                    tool_name = execution.get('tool_name', 'unknown')
                    tool_usage_count[tool_name] = tool_usage_count.get(tool_name, 0) + 1
                    
            except Exception as e:
                print(f"Warning: Failed to read tool log {log_file}: {e}")
        
        # 使用頻度順にソート
        sorted_tools = sorted(tool_usage_count.items(), key=lambda x: x[1], reverse=True)
        
        tool_usage_text = "\n".join([f"  - {tool}: {count} times" for tool, count in sorted_tools])
        
        return f"""
- **Total Tool Executions**: {total_executions}
- **Successful**: {successful_executions}
- **Failed**: {failed_executions}
- **Total Execution Time**: {total_time:.2f} seconds
- **Tool Usage**:
{tool_usage_text}
"""
    
    def _format_orchestration_decisions(self, decisions: List[Dict[str, Any]]) -> str:
        """Format orchestration decisions for markdown"""
        if not decisions:
            return "No orchestration decisions recorded."
        
        formatted = []
        for i, decision in enumerate(decisions, 1):
            formatted.append(f"""
### Decision {i}
- **Step**: {decision.get('step', 'N/A')}
- **Chosen Agent**: {decision.get('chosen_agent', 'N/A')}
- **Reasoning**: {decision.get('reasoning', 'N/A')}
- **Context Considered**: {decision.get('context_summary', 'N/A')}
""")
        
        return "\n".join(formatted)
    
    def _format_budget_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format budget analysis for markdown"""
        if not analysis:
            return "No budget analysis available."
        
        return f"""
- **Total Estimated Cost**: ${analysis.get('total_cost', 'N/A')} USD
- **Budget Utilization**: {analysis.get('budget_utilization', 'N/A')}%
- **Cost Breakdown**:
  - Accommodation: ${analysis.get('accommodation_cost', 'N/A')} USD
  - Transportation: ${analysis.get('transportation_cost', 'N/A')} USD
  - Activities: ${analysis.get('activities_cost', 'N/A')} USD
  - Food: ${analysis.get('food_cost', 'N/A')} USD
"""
    
    def _format_agent_execution_details(self, plan_data: Dict[str, Any]) -> str:
        """Format detailed agent execution information"""
        completed_agents = plan_data.get('completed_agents', [])
        if not completed_agents:
            return "No agent execution details available."
        
        details = []
        for agent in completed_agents:
            details.append(f"- **{agent}**: Completed successfully")
        
        # Add any failed or pending agents if tracked
        total_agents = ['google_maps_agent', 'plan_agent', 'route_agent', 'accommodation_agent', 'summary_agent']
        pending_agents = [agent for agent in total_agents if agent not in completed_agents]
        
        if pending_agents:
            details.append("\n### Pending/Incomplete Agents")
            for agent in pending_agents:
                details.append(f"- **{agent}**: Not executed")
        
        return "\n".join(details)
    
    def _generate_token_usage_summary(self) -> str:
        """Generate token usage summary from saved file"""
        token_path = self.output_dir / "token_usage_summary.json"
        
        if not token_path.exists():
            return "No token usage data available."
        
        try:
            with open(token_path, 'r', encoding='utf-8') as f:
                token_data = json.load(f)
            
            total_usage = token_data.get('total_usage', {})
            agent_usage = token_data.get('token_usage_by_agent', {})
            
            # Format agent usage
            agent_usage_text = []
            for agent_name, usage in agent_usage.items():
                agent_usage_text.append(f"""  - **{agent_name}**:
    - Invocations: {usage.get('invocations', 0)}
    - Input tokens: {usage.get('total_input_tokens', 0):,}
    - Output tokens: {usage.get('total_output_tokens', 0):,}
    - Total tokens: {usage.get('total_tokens', 0):,}
    - Cost: ${usage.get('total_cost_usd', 0):.4f}""")
            
            return f"""
### Total Token Usage
- **Total Input Tokens**: {total_usage.get('total_input_tokens', 0):,}
- **Total Output Tokens**: {total_usage.get('total_output_tokens', 0):,}
- **Total Tokens**: {total_usage.get('total_tokens', 0):,}
- **Total Cost**: ${total_usage.get('total_cost_usd', 0):.4f}

### Usage by Agent
{chr(10).join(agent_usage_text)}
"""
        except Exception as e:
            return f"Error reading token usage data: {str(e)}"
    
    def save_token_usage(self, token_summary: Dict[str, Any]):
        """Save token usage summary"""
        token_path = self.output_dir / "token_usage_summary.json"
        with open(token_path, 'w', encoding='utf-8') as f:
            json.dump(token_summary, f, indent=2, ensure_ascii=False)
    
    def save_video_analysis(self, video_analysis: Dict[str, Any]):
        """Save video analysis results"""
        video_path = self.output_dir / "video_analysis.json"
        with open(video_path, 'w', encoding='utf-8') as f:
            json.dump(video_analysis, f, indent=2, ensure_ascii=False)
    
    def generate_experiment_summary(self) -> Dict[str, str]:
        """Generate experiment summary with file paths"""
        return {
            "experiment_name": self.experiment_name,
            "output_directory": str(self.output_dir),
            "config_file": str(self.output_dir / "experiment_config.json"),
            "final_plan_json": str(self.output_dir / "final_travel_plan.json"),
            "travel_guide": str(self.output_dir / "reports" / "travel_guide.md"),
            "execution_log": str(self.output_dir / "reports" / "execution_log.md"),
            "orchestration_log": str(self.output_dir / "orchestration_log.json"),
            "raw_data_directory": str(self.output_dir / "raw_data"),
            "tool_logs_directory": str(self.output_dir / "tool_logs"),  # ツールログディレクトリを追加
            "token_usage_summary": str(self.output_dir / "token_usage_summary.json")
        } 