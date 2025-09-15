import os
import sys
import json
import pickle
import argparse
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from langchain.schema import HumanMessage

from agents import create_agents
from model_config import ModelConfig, print_model_configuration_help
from prompt import (
    AUTONOMOUS_ORCHESTRATOR_PROMPT, 
    BUDGET_ANALYSIS_PROMPT,
    ORCHESTRATION_DECISION_PROMPT,
    LOCATION_DETERMINATION_PROMPT
)
from output_manager import OutputManager
from tools import tool_logger  # Import tool logger
from token_counter import token_counter
from video_analyzer import VideoAnalyzer

# Load environment variables
load_dotenv()

class AutonomousOrchestratorAgent:
    def __init__(self, model_type: str = 'gemini', model_name: str = None, 
                 experiment_name: str = None, temperature: float = 0.7):
        """
        Initialize method

        Args:
            model_type: Model type to use ('gemini' or 'openai')
            model_name: Specific model name (defaults if omitted)
            experiment_name: Experiment name
            temperature: LLM temperature setting
        """
        # Validate model configuration
        validation = ModelConfig.validate_model_configuration(model_type, model_name)
        
        if validation['warnings']:
            for warning in validation['warnings']:
                print(f"⚠️ {warning}")
        
        if not validation['valid']:
            for error in validation['errors']:
                print(f"❌ {error}")
            raise ValueError("Invalid model configuration")
        
        # Create LLM instance
        self.llm = ModelConfig.create_llm(
            model_type=model_type,
            model_name=model_name,
            temperature=temperature
        )
        
        # Save model information
        self.model_info = validation['model_info']
        
        print(f"🤖 Using model: {self.model_info['model_name']} ({self.model_info['provider']})")
        
        # Create all agents
        self.agents = create_agents(self.llm)
        self.orchestration_decisions = []
        self.shared_context = {}
        self.completed_tasks = set()
        
        # Initialize output manager
        self.output_manager = OutputManager(experiment_name)
        
        # Save model configuration to experiment
        self.output_manager.save_model_config(self.model_info)
        
        # Available agent names for orchestration
        self.available_agents = [
            'google_map_agent',
            'plan_agent', 
            'route_search_agent',
            'accommodation_agent',
            'summary_agent'
        ]
    
    def orchestrate_autonomous_planning(self, user_request: Dict[str, Any]) -> Dict[str, str]:
        """
        Autonomous orchestration method that dynamically determines agent execution
        """
        # Determine location if not provided
        if not user_request.get('location_context') or user_request.get('location_context') == 'auto':
            location_result = self._determine_location(user_request)
            if location_result['success']:
                user_request['location_context'] = location_result['location']
                print(f"📍 Auto-detected location: {location_result['location']} (confidence: {location_result['confidence']})")
            else:
                print(f"❌ Failed to auto-detect location: {location_result['error']}")
                return {'error': 'Failed to determine location automatically'}
        
        # Save experiment configuration
        self.output_manager.save_experiment_config(user_request)
        
        # Initialize shared context from user request
        self.shared_context.update(user_request)
        
        # Extract constraints with specific USD budget
        constraints = {
            'people_count': user_request.get('people_count', 1),
            'days': user_request.get('days', 3),
            'budget': float(user_request.get('budget_usd', 3000))  # Specific USD amount
        }
        
        print(f"🤖 Starting autonomous travel planning...")
        print(f"📍 Location: {user_request['location_context']}")
        print(f"💰 Budget: ${constraints['budget']} USD for {constraints['people_count']} people {constraints['days']} days")
        
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        try:
            while iteration < max_iterations:
                iteration += 1
                print(f"\n🔄 Orchestration iteration {iteration}")
                
                # Make autonomous decision about next action
                decision = self._make_orchestration_decision(constraints)
                self.orchestration_decisions.append({
                    **decision,
                    'iteration': iteration,
                    'timestamp': self._get_timestamp()
                })
                
                # Save orchestration decision
                self.output_manager.save_orchestration_log(self.orchestration_decisions)
                
                print(f"🎯 Decision: {decision.get('action', 'unknown')}")
                if decision.get('chosen_agent'):
                    print(f"🤖 Selected agent: {decision['chosen_agent']}")
                print(f"💭 Reasoning: {decision.get('reasoning', 'No reasoning provided')}")
                
                # Execute decision
                if decision['action'] == 'complete_planning':
                    print("✅ Autonomous orchestration completed!")
                    break
                elif decision['action'] == 'call_agent':
                    agent_name = decision['chosen_agent']
                    task_description = decision['task_description']
                    
                    # Execute chosen agent
                    result = self._execute_agent(agent_name, task_description, constraints)
                    
                    # Save agent result
                    self.output_manager.save_agent_result(agent_name, result)
                    
                    # Mark task as completed
                    self.completed_tasks.add(agent_name)
                else:
                    print(f"⚠️ Unknown action: {decision['action']}")
                    break
            
            # Generate final comprehensive plan
            final_plan_data = self._compile_final_results(constraints)
            
            # Generate budget analysis
            budget_analysis = self._generate_budget_analysis(final_plan_data, constraints)
            final_plan_data['budget_analysis'] = budget_analysis
            
            # Save final plan
            self.output_manager.save_final_plan(final_plan_data)
            
            # Save token usage summary
            token_summary = token_counter.get_summary()
            self.output_manager.save_token_usage(token_summary)
            
            # Print token usage summary
            token_counter.print_summary()
            
            # Generate experiment summary
            experiment_summary = self.output_manager.generate_experiment_summary()
            
            print(f"\n🎉 Experiment completed successfully!")
            print(f"📁 Results saved to: {experiment_summary['output_directory']}")
            
            return experiment_summary
            
        except Exception as e:
            print(f"\n❌ Autonomous orchestration failed: {str(e)}")
            error_summary = {
                'error': str(e),
                'iteration': iteration,
                'completed_tasks': list(self.completed_tasks),
                'output_directory': str(self.output_manager.output_dir)
            }
            return error_summary
    
    def _make_orchestration_decision(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Use prompts to autonomously determine next action"""
        
        # Prepare context for decision making
        context_summary = {
            'available_info': self._summarize_available_info(),
            'completed_tasks': list(self.completed_tasks),
            'remaining_constraints': constraints,
            'shared_context_keys': list(self.shared_context.keys())
        }
        
        previous_actions = [
            {
                'agent': decision.get('chosen_agent'),
                'task': decision.get('task_description'),
                'iteration': decision.get('iteration')
            } for decision in self.orchestration_decisions[-3:]  # Last 3 decisions
        ]
        
        # Use the proper orchestration prompt from prompt.py
        prompt = AUTONOMOUS_ORCHESTRATOR_PROMPT.format(
            context=json.dumps(context_summary, indent=2),
            available_agents=json.dumps(self.available_agents),
            constraints=json.dumps(constraints),
            previous_actions=json.dumps(previous_actions, indent=2)
        )
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Track token usage for orchestrator with response metadata
            input_tokens = None
            output_tokens = None
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                if 'usage_metadata' in metadata:
                    usage = metadata['usage_metadata']
                    input_tokens = usage.get('prompt_token_count')
                    output_tokens = usage.get('candidates_token_count')
                elif 'token_usage' in metadata:
                    usage = metadata['token_usage']
                    input_tokens = usage.get('prompt_tokens')
                    output_tokens = usage.get('completion_tokens')
            
            token_counter.track_usage(
                "Orchestrator", 
                self.model_info['model_name'], 
                prompt, 
                response.content,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            
            # Parse JSON response
            decision_text = response.content.strip()
            if decision_text.startswith('```json'):
                decision_text = decision_text.split('```json')[1].split('```')[0].strip()
            elif decision_text.startswith('```'):
                decision_text = decision_text.split('```')[1].split('```')[0].strip()
            
            decision = json.loads(decision_text)
            
            # Validate decision
            if decision.get('action') not in ['call_agent', 'complete_planning']:
                decision['action'] = 'complete_planning'
            
            if decision['action'] == 'call_agent' and decision.get('chosen_agent') not in self.available_agents:
                decision['action'] = 'complete_planning'
                decision['chosen_agent'] = None
            
            return decision
            
        except Exception as e:
            print(f"⚠️ Orchestration decision error: {str(e)}")
            # Fallback decision logic
            return self._fallback_decision_logic()
    
    def _fallback_decision_logic(self) -> Dict[str, Any]:
        """Fallback logic when LLM decision fails"""
        
        if 'google_map_agent' not in self.completed_tasks:
            return {
                'action': 'call_agent',
                'chosen_agent': 'google_map_agent',
                'task_description': 'Collect detailed POI information',
                'reasoning': 'Fallback: POI information needed first',
                'context_summary': 'Fallback decision'
            }
        elif 'plan_agent' not in self.completed_tasks:
            return {
                'action': 'call_agent',
                'chosen_agent': 'plan_agent',
                'task_description': 'Create travel plan based on constraints',
                'reasoning': 'Fallback: Planning needed',
                'context_summary': 'Fallback decision'
            }
        elif 'route_search_agent' not in self.completed_tasks:
            return {
                'action': 'call_agent',
                'chosen_agent': 'route_search_agent',
                'task_description': 'Search routes between POIs',
                'reasoning': 'Fallback: Route information needed',
                'context_summary': 'Fallback decision'
            }
        elif 'accommodation_agent' not in self.completed_tasks:
            return {
                'action': 'call_agent',
                'chosen_agent': 'accommodation_agent',
                'task_description': 'Search for accommodations',
                'reasoning': 'Fallback: Accommodation information needed',
                'context_summary': 'Fallback decision'
            }
        elif 'summary_agent' not in self.completed_tasks:
            return {
                'action': 'call_agent',
                'chosen_agent': 'summary_agent',
                'task_description': 'Create final comprehensive travel plan',
                'reasoning': 'Fallback: Final summary needed',
                'context_summary': 'Fallback decision'
            }
        else:
            return {
                'action': 'complete_planning',
                'chosen_agent': None,
                'task_description': 'Planning completed',
                'reasoning': 'All agents executed',
                'context_summary': 'Completion state'
            }

    def _execute_agent(self, agent_name: str, task_description: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific agent with given context"""
        try:
            agent = self.agents.get(agent_name)
            if not agent:
                return {
                    'agent': agent_name,
                    'task_completed': False,
                    'error': f'Agent {agent_name} not found'
                }
            
            # Prepare context for agent
            context = self._prepare_agent_context(agent_name, constraints)
            
            print(f"🔧 Executing {agent_name}...")
            print(f"📋 Task: {task_description}")
            
            # Clear tool logs (starting new agent execution)
            tool_logger.clear_logs()
            
            # Execute agent
            result = agent.execute(task_description, context)
            
            # Save tool logs after agent execution
            if tool_logger.execution_logs:
                self.output_manager.save_tool_logs(tool_logger.execution_logs, agent_name)
                print(f"🔧 Saved tool execution logs for {agent_name}")
            
            # Update shared context with results
            self._update_shared_context(agent_name, result)
            
            if result.get('task_completed', False):
                print(f"✅ {agent_name} execution completed")
            else:
                print(f"⚠️ {agent_name} execution had issues: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error occurred during {agent_name} execution: {str(e)}")
            
            # Save tool logs even on error
            if tool_logger.execution_logs:
                self.output_manager.save_tool_logs(tool_logger.execution_logs, agent_name)
            
            return {
                'agent': agent_name,
                'task_completed': False,
                'error': str(e)
            }

    def _prepare_agent_context(self, agent_name: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare appropriate context for agent"""
        base_context = {
            'people_count': constraints['people_count'],
            'days': constraints['days'],
            'budget': constraints['budget'],
            **self.shared_context
        }
        
        # Add video insights if available
        if 'video_insights' in self.shared_context:
            base_context['video_insights'] = self.shared_context['video_insights']
        
        # Agent-specific context preparation
        if agent_name == 'google_map_agent':
            return {
                **base_context,
                'poi_list': self.shared_context.get('poi_list', []),
                'location_context': self.shared_context.get('location_context', '')
            }
        elif agent_name == 'plan_agent':
            return {
                **base_context,
                'poi_data': self.shared_context.get('poi_data', []),
                'route_info': self.shared_context.get('route_info', {})
            }
        elif agent_name == 'route_search_agent':
            return {
                **base_context,
                'selected_pois': self.shared_context.get('selected_pois', []),
                'transport_prefs': self.shared_context.get('transport_prefs', 'public_transport')
            }
        elif agent_name == 'accommodation_agent':
            return {
                **base_context,
                'itinerary': self.shared_context.get('itinerary', ''),
                'location_context': self.shared_context.get('location_context', ''),
                'preferences': self.shared_context.get('accommodation_preferences', {}),
                'selected_pois': self.shared_context.get('selected_pois', [])
            }
        elif agent_name == 'summary_agent':
            return {
                **base_context,
                'poi_data': self.shared_context.get('poi_data', []),
                'itinerary': self.shared_context.get('itinerary', ''),
                'routes': self.shared_context.get('routes', []),
                'accommodation_options': self.shared_context.get('accommodation_options', []),
                'recommended_hotel': self.shared_context.get('recommended_hotel', None),
                'restaurant_recommendations': self.shared_context.get('restaurant_recommendations', {})
            }
        else:
            return base_context

    def _update_shared_context(self, agent_name: str, result: Dict[str, Any]):
        """Update shared context with agent results"""
        if not result.get('task_completed', False):
            return
            
        if agent_name == 'google_map_agent':
            self.shared_context['poi_data'] = result.get('poi_data', [])
            self.shared_context['poi_analysis'] = result.get('analysis', '')
        elif agent_name == 'plan_agent':
            self.shared_context['itinerary'] = result.get('itinerary', '')
            self.shared_context['selected_pois'] = result.get('selected_pois', [])
            self.shared_context['restaurant_recommendations'] = result.get('restaurant_recommendations', {})
        elif agent_name == 'route_search_agent':
            self.shared_context['routes'] = result.get('routes', [])
            self.shared_context['transport_analysis'] = result.get('transport_analysis', '')
        elif agent_name == 'accommodation_agent':
            self.shared_context['accommodation_options'] = result.get('accommodation_search_results', [])
            self.shared_context['accommodation_analysis'] = result.get('accommodation_analysis', '')
            self.shared_context['recommended_hotel'] = result.get('recommended_hotel', None)
        elif agent_name == 'summary_agent':
            self.shared_context['comprehensive_summary'] = result.get('comprehensive_summary', '')
            self.shared_context['detailed_budget_analysis'] = result.get('detailed_budget_analysis', '')

    def _summarize_available_info(self) -> Dict[str, bool]:
        """Get summary of available information"""
        return {
            'poi_data': 'poi_data' in self.shared_context,
            'itinerary': 'itinerary' in self.shared_context,
            'routes': 'routes' in self.shared_context,
            'accommodation': 'accommodation_options' in self.shared_context,
            'final_summary': 'comprehensive_summary' in self.shared_context
        }

    def _compile_final_results(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Compile final results"""
        return {
            'experiment_constraints': constraints,
            'model_info': self.model_info,
            'poi_data': self.shared_context.get('poi_data', []),
            'selected_pois': self.shared_context.get('selected_pois', []),
            'itinerary': self.shared_context.get('itinerary', ''),
            'routes': self.shared_context.get('routes', []),
            'accommodation_options': self.shared_context.get('accommodation_options', []),
            'recommended_hotel': self.shared_context.get('recommended_hotel', None),
            'restaurant_recommendations': self.shared_context.get('restaurant_recommendations', {}),
            'comprehensive_summary': self.shared_context.get('comprehensive_summary', ''),
            'detailed_budget_analysis': self.shared_context.get('detailed_budget_analysis', ''),
            'orchestration_decisions': self.orchestration_decisions,
            'total_iterations': len(self.orchestration_decisions),
            'completed_agents': list(self.completed_tasks)
        }

    def _generate_budget_analysis(self, plan_data: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed budget analysis using prompt.py"""
        try:
            # Use budget analysis prompt from prompt.py
            budget_prompt = BUDGET_ANALYSIS_PROMPT.format(
                total_budget=constraints['budget'],
                people_count=constraints['people_count'],
                days=constraints['days'],
                selected_pois=json.dumps([poi.get('name', '') for poi in plan_data.get('selected_pois', [])]),
                transportation=json.dumps(plan_data.get('routes', []), indent=2),
                accommodation=json.dumps(plan_data.get('accommodation_options', [])[:1], indent=2)
            )
            
            response = self.llm.invoke([HumanMessage(content=budget_prompt)])
            
            # Track token usage for budget analysis with response metadata
            input_tokens = None
            output_tokens = None
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                if 'usage_metadata' in metadata:
                    usage = metadata['usage_metadata']
                    input_tokens = usage.get('prompt_token_count')
                    output_tokens = usage.get('candidates_token_count')
                elif 'token_usage' in metadata:
                    usage = metadata['token_usage']
                    input_tokens = usage.get('prompt_tokens')
                    output_tokens = usage.get('completion_tokens')
            
            token_counter.track_usage(
                "Orchestrator-BudgetAnalysis", 
                self.model_info['model_name'], 
                budget_prompt, 
                response.content,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            
            # Also calculate estimated costs
            cost_estimates = self._estimate_costs(plan_data, constraints)
            
            return {
                'llm_budget_analysis': response.content,
                'cost_estimates': cost_estimates,
                'budget_utilization': {
                    'total_budget': constraints['budget'],
                    'estimated_total_cost': sum(cost_estimates.values()),
                    'utilization_percentage': (sum(cost_estimates.values()) / constraints['budget']) * 100 if constraints['budget'] > 0 else 0
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'cost_estimates': self._estimate_costs(plan_data, constraints)
            }

    def _estimate_costs(self, plan_data: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, float]:
        """Cost estimation (simple calculation)"""
        budget = constraints['budget']
        people_count = constraints['people_count']
        days = constraints['days']
        
        # Simple cost allocation
        accommodation_cost = budget * 0.4  # 40% for accommodation
        transportation_cost = budget * 0.2  # 20% for transportation
        food_cost = budget * 0.25  # 25% for food
        activity_cost = budget * 0.15  # 15% for activities
        
        return {
            'accommodation': accommodation_cost,
            'transportation': transportation_cost,
            'food': food_cost,
            'activities': activity_cost
        }

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def extract_poi_from_graph(self, video_id: str) -> Dict[str, Any]:
        """
        Extract POI list from graph.pkl file using video_id

        Args:
            video_id: YouTube video ID

        Returns:
            Dict containing:
                - success: bool
                - poi_list: List[str] POI name list
                - location: str Inferred location
                - error: str (if failed)
        """
        try:
            # Construct path to graph.pkl file
            graph_path = os.path.join('data', 'graphs_v2', video_id, 'graph.pkl')
            
            if not os.path.exists(graph_path):
                return {
                    'success': False,
                    'error': f'Graph file not found: {graph_path}'
                }
            
            # Add path to graph/utils module
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            graph_dir = os.path.join(parent_dir, 'graph')
            if graph_dir not in sys.path:
                sys.path.insert(0, graph_dir)
            
            # Create dummy classes (if lingua dependency is missing)
            try:
                import utils
            except ImportError as e:
                if 'lingua' in str(e):
                    # Create dummy lingua module
                    import types
                    lingua = types.ModuleType('lingua')
                    
                    class DummyLanguageDetectorBuilder:
                        @classmethod
                        def from_all_languages(cls):
                            return cls()
                        @classmethod
                        def from_languages(cls, *languages):
                            return cls()
                        def build(self):
                            return DummyLanguageDetector()
                    
                    class DummyLanguageDetector:
                        def detect_language_of(self, text):
                            return None
                    
                    class DummyLanguage:
                        ENGLISH = 'en'
                        JAPANESE = 'ja'
                    
                    lingua.LanguageDetectorBuilder = DummyLanguageDetectorBuilder
                    lingua.Language = DummyLanguage
                    sys.modules['lingua'] = lingua
                    
                    # Re-import utils
                    import utils
                else:
                    raise
            
            # Load graph.pkl
            with open(graph_path, 'rb') as f:
                graph = pickle.load(f)
            
            # Extract POIs
            poi_list = []
            location_candidates = []
            
            for node_id, node in graph.nodes.items():
                node_type = str(node.node_type).split('.')[-1]  # NodeType.POI -> POI
                
                if node_type == 'POI':
                    # Prioritize Japanese name, fall back to English
                    name = node.display_name.get('ja') or node.display_name.get('en', '')
                    if name:
                        poi_list.append(name)
                elif node_type == 'CITY':
                    city_name = node.display_name.get('ja') or node.display_name.get('en', '')
                    if city_name:
                        location_candidates.append(city_name)
                elif node_type == 'PREFECTURE':
                    prefecture_name = node.display_name.get('ja') or node.display_name.get('en', '')
                    if prefecture_name:
                        location_candidates.append(prefecture_name)
            
            # Infer location (prioritize city)
            location = location_candidates[0] if location_candidates else 'auto'
            
            print(f"📍 Extracted {len(poi_list)} POIs from {video_id}")
            if location != 'auto':
                print(f"📍 Inferred location: {location}")
            
            return {
                'success': True,
                'poi_list': poi_list,
                'location': location,
                'video_id': video_id
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _determine_location(self, user_request: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically determine location using LLM"""
        try:
            # Prepare context for location determination
            poi_list = user_request.get('poi_list', [])
            video_insights = user_request.get('video_insights', '')
            user_input = user_request.get('user_input', '')
            
            # Use location determination prompt
            prompt = LOCATION_DETERMINATION_PROMPT.format(
                poi_list=json.dumps(poi_list) if poi_list else 'None',
                video_insights=video_insights if video_insights else 'None',
                user_input=user_input if user_input else 'None'
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Track token usage
            input_tokens = None
            output_tokens = None
            if hasattr(response, 'response_metadata'):
                metadata = response.response_metadata
                if 'usage_metadata' in metadata:
                    usage = metadata['usage_metadata']
                    input_tokens = usage.get('prompt_token_count')
                    output_tokens = usage.get('candidates_token_count')
                elif 'token_usage' in metadata:
                    usage = metadata['token_usage']
                    input_tokens = usage.get('prompt_tokens')
                    output_tokens = usage.get('completion_tokens')
            
            token_counter.track_usage(
                "Orchestrator-LocationDetermination", 
                self.model_info['model_name'], 
                prompt, 
                response.content,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            
            # Parse JSON response
            location_text = response.content.strip()
            if location_text.startswith('```json'):
                location_text = location_text.split('```json')[1].split('```')[0].strip()
            elif location_text.startswith('```'):
                location_text = location_text.split('```')[1].split('```')[0].strip()
            
            location_data = json.loads(location_text)
            
            return {
                'success': True,
                'location': location_data.get('location', 'Tokyo, Japan'),
                'confidence': location_data.get('confidence', 'low'),
                'reasoning': location_data.get('reasoning', ''),
                'alternatives': location_data.get('alternative_locations', [])
            }
            
        except Exception as e:
            print(f"⚠️ Location determination error: {str(e)}")
            # Fallback to default location
            return {
                'success': False,
                'error': str(e),
                'location': 'Tokyo, Japan',
                'confidence': 'low',
                'reasoning': 'Failed to determine location, using default'
            }

def create_argument_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='Multi-agent travel planning system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Specify POI list (traditional method)
  python main.py --poi-list "Tokyo Tower" "Senso-ji Temple" "Shibuya Crossing"
  
  # Use POI only from video_id
  python main.py --video-id UU8YnZ7iBOw --use-poi
  
  # Use video only from video_id (Gemini model required)
  python main.py --video-id UU8YnZ7iBOw --use-video --model-type gemini
  
  # Use both POI and video from video_id (Gemini model required)
  python main.py --video-id UU8YnZ7iBOw --use-poi --use-video --model-type gemini
  
  # For video_id starting with hyphen (use equals sign)
  python main.py --video-id="-33HgQh7QEU" --use-poi --use-video --model-type gemini
  
  # Extract POI from video_id and also specify additional POI
  python main.py --video-id UU8YnZ7iBOw --use-poi --poi-list "Additional Restaurant"
  
  # Extract POI from video (auto-detect location)
  python main.py --video-path travel_vlog.mp4
  
  # Extract POI from video (specify location)
  python main.py --video-path travel_vlog.mp4 --location "Tokyo, Japan"
  
  # Combination of video and POI list
  python main.py --video-path vlog.mp4 --poi-list "Must-visit Restaurant" --location "Kyoto"
  
  # Use specific Gemini model
  python main.py --video-id JDj0WQITs2k --model-type gemini --model-name gemini-pro
  
  # Use OpenAI GPT-4 (auto-detect location)
  python main.py --poi-list "Eiffel Tower" "Louvre" --model-type openai --model-name gpt-4o
  
  # Display supported models
  python main.py --list-models
"""
    )
    
    parser.add_argument(
        '--model-type', 
        choices=['gemini', 'openai'],
        default='gemini',
        help='LLM provider to use (default: gemini)'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        help='Specific model name to use (defaults to provider default if omitted)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='LLM temperature setting (0.0-1.0, default: 0.7)'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        help='Experiment name (auto-generated if omitted)'
    )
    
    parser.add_argument(
        '--poi-list',
        nargs='+',
        default=None,
        help='POI list (space-separated)'
    )
    
    parser.add_argument(
        '--video-path',
        type=str,
        default=None,
        help='Path to travel video (MP4) to analyze'
    )
    
    parser.add_argument(
        '--video-id',
        type=str,
        default=None,
        help='YouTube video ID (specify usage with --use-poi/--use-video flags)'
    )
    
    parser.add_argument(
        '--location',
        type=str,
        default='auto',
        help='Travel destination location ("auto" for auto-detection, default: auto)'
    )
    
    parser.add_argument(
        '--use-poi',
        action='store_true',
        default=False,
        help='Use POI list (only valid with video_id)'
    )
    
    parser.add_argument(
        '--use-video',
        action='store_true',
        default=False,
        help='Use video (only valid with video_id, Gemini model required)'
    )
    
    parser.add_argument(
        '--people-count',
        type=int,
        default=2,
        help='Number of people'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=3,
        help='Number of days'
    )
    
    parser.add_argument(
        '--budget-usd',
        type=float,
        default=2500.0,
        help='Budget (USD)'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='Display list of supported models'
    )
    
    return parser

def main():
    """
    Main function to run the autonomous travel planning system
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # If model list display is requested
    if args.list_models:
        print_model_configuration_help()
        return
    
    # Validate temperature setting
    if not 0.0 <= args.temperature <= 1.0:
        print("❌ Temperature must be between 0.0 and 1.0")
        return
    
    # Validate input: must have either POI list, video, or video_id
    if not args.poi_list and not args.video_path and not args.video_id:
        print("❌ Error: Please specify either POI list (--poi-list), video (--video-path), or video_id (--video-id)")
        return
    
    # Validate video_id usage flags
    if args.video_id and not args.use_poi and not args.use_video:
        print("❌ Error: When using video_id, please specify --use-poi or --use-video flag")
        print("💡 Hint: Add --use-poi to use POIs, --use-video to use video")
        return
    
    # Validation will be done after video_id processing
    
    print(f"🚀 Starting multi-agent travel planning system")
    if args.location != 'auto':
        print(f"📍 Location: {args.location}")
    else:
        print(f"📍 Location: Auto-detect")
    print(f"👥 People: {args.people_count}")
    print(f"📅 Days: {args.days}")
    print(f"💰 Budget: ${args.budget_usd} USD")
    
    try:
        # Create orchestrator with model configuration
        orchestrator = AutonomousOrchestratorAgent(
            model_type=args.model_type,
            model_name=args.model_name,
            experiment_name=args.experiment_name,
            temperature=args.temperature
        )
        
        # Initialize variables for POI list and location
        extracted_poi_list = []
        extracted_location = None
        
        # Process video_id if provided
        if args.video_id:
            # Extract POI from graph.pkl if --use-poi flag is set
            if args.use_poi:
                print(f"📊 Extracting POIs from video_id: {args.video_id}")
                graph_result = orchestrator.extract_poi_from_graph(args.video_id)
                
                if graph_result['success']:
                    extracted_poi_list = graph_result['poi_list']
                    extracted_location = graph_result['location']
                    print(f"✅ Extracted {len(extracted_poi_list)} POIs")
                    
                    # Save extracted POI info
                    orchestrator.output_manager.save_video_analysis({
                        'video_id': args.video_id,
                        'source': 'graph_pkl',
                        'extracted_poi_list': extracted_poi_list,
                        'extracted_location': extracted_location,
                        'poi_count': len(extracted_poi_list)
                    })
                else:
                    print(f"❌ Failed to extract POIs: {graph_result.get('error', 'Unknown error')}")
                    return
            
            # Load video file if --use-video flag is set
            if args.use_video:
                video_file_path = os.path.join('data', 'videos', f'{args.video_id}.mp4')
                if os.path.exists(video_file_path):
                    print(f"🎬 Loading video file for video_id: {video_file_path}")
                    args.video_path = video_file_path
                else:
                    print(f"❌ Video file not found: {video_file_path}")
                    return
        
        # Now validate video usage with model type
        if args.video_path and args.model_type != 'gemini':
            print("❌ Error: Video analysis is only supported with Gemini models")
            print("💡 Hint: Please use --model-type gemini")
            return
        
        # Process video if provided
        if args.video_path:
            print(f"🎥 Uploading video: {args.video_path}")
            video_analyzer = VideoAnalyzer()
            video_result = video_analyzer.upload_video(args.video_path)
            
            if video_result['success']:
                # Store video reference in shared context for agents
                orchestrator.shared_context['video_file'] = video_result.get('video_file')
                orchestrator.shared_context['video_file_name'] = video_result.get('video_file_name')
                orchestrator.shared_context['video_file_uri'] = video_result.get('video_file_uri')
                orchestrator.shared_context['has_video'] = True
                
                print("✅ Video upload completed")
                print("🤖 Agents will directly analyze the video to create a travel plan")
                
                # Save video upload info
                video_info = {
                    "video_path": args.video_path,
                    "upload_status": "success",
                    "video_file_name": video_result.get('video_file_name'),
                    "video_file_uri": video_result.get('video_file_uri')
                }
                orchestrator.output_manager.save_video_analysis(video_info)
            else:
                print(f"❌ Video upload failed: {video_result.get('error', 'Unknown error')}")
                return
        
        # Combine POI lists from different sources
        final_poi_list = []
        
        # Only process POI if not in video-only mode
        if not (args.video_id and args.use_video and not args.use_poi):
            # Add extracted POIs from graph.pkl
            if extracted_poi_list:
                final_poi_list.extend(extracted_poi_list)
            
            # Add user-provided POIs (if not using video_path)
            if args.poi_list and not args.video_path:
                final_poi_list.extend(args.poi_list)
                print(f"🎯 Additional POIs: {', '.join(args.poi_list)}")
            
            # Remove duplicates while preserving order
            seen = set()
            final_poi_list = [x for x in final_poi_list if not (x in seen or seen.add(x))]
            
            if final_poi_list:
                print(f"🎯 Total POI list: {', '.join(final_poi_list[:10])}{'...' if len(final_poi_list) > 10 else ''}")
        
        # Determine final location
        final_location = args.location
        if args.location == 'auto' and extracted_location and extracted_location != 'auto':
            final_location = extracted_location
            print(f"📍 Using location inferred from graph.pkl: {final_location}")
        
        # Construct user request
        user_request = {
            'location_context': final_location,
            'people_count': args.people_count,
            'days': args.days,
            'budget_usd': args.budget_usd,
            'input_source': 'video' if args.video_path else ('graph' if args.video_id else 'manual')
        }
        
        # Include POI list if available
        # When using video_id with both --use-poi and --use-video, include POI list to prevent re-extraction
        if final_poi_list:
            user_request['poi_list'] = final_poi_list
        
        # Add video_id to context if used
        if args.video_id:
            user_request['video_id'] = args.video_id
        
        # Run autonomous planning
        result = orchestrator.orchestrate_autonomous_planning(user_request)
        
        # Print summary
        print(f"\n🎉 Experiment completed!")
        print(f"📁 Results: {result}")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print(f"💡 Hint: Check supported models with --list-models")

if __name__ == "__main__":
    main()