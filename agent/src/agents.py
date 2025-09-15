import json
from typing import Dict, List, Any, Union
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from tools import create_tools, tool_logger
from token_counter import token_counter
import google.generativeai as genai

# Import all prompts from prompt.py
from prompt import (
    GOOGLE_MAP_AGENT_PROMPT,
    PLAN_AGENT_PROMPT,
    ROUTE_SEARCH_AGENT_PROMPT,
    ACCOMMODATION_AGENT_PROMPT,
    SUMMARY_AGENT_PROMPT,
    BUDGET_ANALYSIS_PROMPT,
    TOOL_USAGE_INSTRUCTIONS
)

# Type alias for supported chat models
SupportedChatModel = Union[ChatGoogleGenerativeAI, ChatOpenAI]

class BaseAgent:
    def __init__(self, llm: SupportedChatModel, tools: List = None):
        self.llm = llm
        self.tools = tools or []
        self.memory = []
        # Get model name for token tracking
        self.model_name = self._get_model_name(llm)
        # Check if using Gemini model
        self.is_gemini = 'gemini' in self.model_name.lower()
    
    def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Base execution method to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement execute method")
    
    def _save_tool_logs(self, output_manager, agent_name: str):
        """Save tool logs"""
        if tool_logger.execution_logs:
            output_manager.save_tool_logs(tool_logger.execution_logs, agent_name)
            tool_logger.clear_logs()  # ログをクリア
    
    def _get_model_name(self, llm: SupportedChatModel) -> str:
        """Get model name from LLM instance"""
        if hasattr(llm, 'model_name'):
            return llm.model_name
        elif hasattr(llm, 'model'):
            return llm.model
        else:
            return "unknown"
    
    def _track_token_usage(self, agent_name: str, input_text: str, output_text: str, response=None):
        """Track token usage for this agent
        
        Args:
            agent_name: Name of the agent
            input_text: Input prompt text
            output_text: Output response text
            response: LangChain response object that may contain usage metadata
        """
        input_tokens = None
        output_tokens = None
        
        # Try to extract token counts from response metadata
        if response and hasattr(response, 'response_metadata'):
            metadata = response.response_metadata
            
            # For Gemini models
            if 'usage_metadata' in metadata:
                usage = metadata['usage_metadata']
                input_tokens = usage.get('prompt_token_count')
                output_tokens = usage.get('candidates_token_count')
            
            # For OpenAI models
            elif 'token_usage' in metadata:
                usage = metadata['token_usage']
                input_tokens = usage.get('prompt_tokens')
                output_tokens = usage.get('completion_tokens')
        
        # Track usage with native token counts if available
        token_counter.track_usage(
            agent_name, 
            self.model_name, 
            input_text, 
            output_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
    
    def _analyze_with_video(self, prompt: str, video_file) -> str:
        """動画を含めてGeminiで分析する
        
        Args:
            prompt: 分析用プロンプト
            video_file: Geminiにアップロードされた動画ファイル
            
        Returns:
            分析結果のテキスト
        """
        if not self.is_gemini:
            # Non-Gemini models can't handle video directly
            return self.llm.invoke([HumanMessage(content=prompt)]).content
        
        try:
            # Use Gemini native client for video analysis
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content([prompt, video_file])
            
            # Track token usage if available
            if hasattr(response, 'usage_metadata'):
                self._track_token_usage(
                    self.name,
                    prompt,
                    response.text,
                    response  # Pass the native response for metadata extraction
                )
            
            return response.text
        except Exception as e:
            print(f"⚠️ 動画分析エラー: {str(e)}")
            # Fallback to text-only analysis
            return self.llm.invoke([HumanMessage(content=prompt)]).content

class GoogleMapAgent(BaseAgent):
    def __init__(self, llm: SupportedChatModel, tools: List):
        # Only use POI-related tools
        relevant_tools = [tool for tool in tools if 'poi' in tool.name.lower() or 'google' in tool.name.lower()]
        super().__init__(llm, relevant_tools)
        self.name = "Google Map Agent"
    
    def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get detailed information for specific POIs using prompt.py prompts"""
        try:
            # Clear tool logs (starting new execution)
            tool_logger.clear_logs()
            
            poi_list = context.get('poi_list', [])
            location_context = context.get('location_context', '')
            video_file = context.get('video_file')
            
            # If video is provided but no POI list, extract POIs from video first
            if video_file and not poi_list:
                print("🎥 動画からPOIを抽出中...")
            elif video_file and poi_list:
                print(f"📊 既存のPOIリスト（{len(poi_list)}個）を使用します。動画からのPOI抽出をスキップします。")
                
                video_analysis_prompt = f"""
Analyze this travel video and extract all Points of Interest (POIs) shown.
Location context: {location_context if location_context else "Not specified"}

Please identify:
1. Tourist attractions, landmarks, restaurants, shops, and activities
2. Both famous landmarks and local spots
3. Any place names shown or mentioned

List the POIs in JSON format:
{{
    "poi_list": ["POI 1 name", "POI 2 name", ...],
    "location_identified": "city/region if identifiable"
}}
"""
                
                try:
                    response_text = self._analyze_with_video(video_analysis_prompt, video_file)
                    if response_text:
                        # Parse JSON response
                        import re
                        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
                        if json_match:
                            video_data = json.loads(json_match.group())
                            poi_list = video_data.get('poi_list', [])
                            if video_data.get('location_identified') and not location_context:
                                location_context = video_data['location_identified']
                            print(f"✅ 動画から {len(poi_list)} 個のPOIを抽出しました")
                except Exception as e:
                    print(f"⚠️ 動画分析エラー: {str(e)}")
                    return {
                        'agent': self.name,
                        'task_completed': False,
                        'error': f'Video analysis failed: {str(e)}',
                        'poi_data': []
                    }
            
            if not poi_list:
                return {
                    'agent': self.name,
                    'task_completed': False,
                    'error': 'No POI list provided or extracted from video',
                    'poi_data': []
                }
            
            # Use the proper prompt from prompt.py
            prompt_text = GOOGLE_MAP_AGENT_PROMPT.format(
                poi_list=json.dumps(poi_list),
                location_context=location_context
            )
            
            # Add tool usage instructions
            full_prompt = f"{prompt_text}\n\n{TOOL_USAGE_INSTRUCTIONS}"
            
            # Get detailed information for each POI using tools
            poi_details_tool = next((tool for tool in self.tools if 'poi_details' in tool.name), None)
            poi_data = []
            
            if poi_details_tool:
                for poi_name in poi_list:
                    # Try search with retry logic
                    poi_info = self._search_poi_with_retry(
                        poi_details_tool, poi_name, location_context
                    )
                    
                    if 'error' not in poi_info:
                        poi_data.append(poi_info)
                    else:
                        # Add minimal info for failed POIs
                        poi_data.append({
                            'name': poi_name,
                            'error': poi_info['error'],
                            'importance_score': 0,
                            'location_context': location_context,
                            'retry_attempts': poi_info.get('retry_attempts', 0)
                        })
            else:
                # No tools available, create basic structure
                for poi_name in poi_list:
                    poi_data.append({
                        'name': poi_name,
                        'error': 'No POI details tool available',
                        'importance_score': 50,  # Default score
                        'location_context': location_context
                    })
            
            # Use LLM with proper prompt for analysis
            analysis_prompt = f"""
            {full_prompt}
            
            Based on the following POI data, provide comprehensive analysis:
            POI Data: {json.dumps(poi_data, indent=2)}
            
            For each POI, analyze:
            1. Importance and visitor appeal
            2. Optimal visit duration recommendations
            3. Best visiting times and crowd considerations
            4. Cost implications and budget considerations
            5. Group logistics and coordination needs
            6. Must-see priority ranking (1-10 scale)
            
            Focus on practical planning information in English.
            """
            
            analysis_response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            
            # Track token usage
            self._track_token_usage(self.name, analysis_prompt, analysis_response.content)
            
            result = {
                'agent': self.name,
                'task_completed': True,
                'poi_data': poi_data,
                'analysis': analysis_response.content,
                'total_pois': len(poi_data),
                'successfully_found': len([poi for poi in poi_data if 'error' not in poi]),
                'tools_used': [tool.name for tool in self.tools],
                'tool_execution_count': len(tool_logger.execution_logs)
            }
            
            self.memory.append(result)
            return result
            
        except Exception as e:
            return {
                'agent': self.name,
                'task_completed': False,
                'error': str(e),
                'poi_data': [],
                'tool_execution_count': len(tool_logger.execution_logs)
            }
    
    def _search_poi_with_retry(self, poi_details_tool, poi_name: str, location_context: str) -> Dict:
        """Search for POI with retry logic and search term variations"""
        max_retries = 3
        retry_count = 0
        
        # Generate search variations
        search_variations = self._generate_search_variations(poi_name, location_context)
        
        for search_query in search_variations:
            retry_count += 1
            print(f"🔍 検索中 (試行 {retry_count}/{max_retries}): {search_query}")
            
            try:
                # Use the tool with the search query
                if '|' in search_query:
                    poi_info = poi_details_tool.func(search_query)
                else:
                    poi_info = poi_details_tool.func(f"{search_query}|{location_context}")
                
                # If successful, return the result
                if 'error' not in poi_info:
                    print(f"✅ 見つかりました: {poi_info.get('name', poi_name)}")
                    return poi_info
                else:
                    print(f"⚠️ 見つかりませんでした: {poi_info.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ 検索エラー: {str(e)}")
            
            if retry_count >= max_retries:
                break
        
        # Return error with retry count
        return {
            'name': poi_name,
            'error': f'POI not found after {retry_count} attempts with variations',
            'retry_attempts': retry_count,
            'search_variations_tried': search_variations[:retry_count]
        }
    
    def _generate_search_variations(self, poi_name: str, location_context: str) -> List[str]:
        """Generate search term variations for better POI discovery"""
        variations = []
        
        # List of invalid location contexts to ignore
        invalid_locations = ['unknown', 'not specified', '', 'cannot be determined', 'undetermined', 'n/a', 'none']
        
        # Check if location context is invalid
        is_location_invalid = (not location_context or 
                              any(invalid in location_context.lower() for invalid in invalid_locations))
        
        # If location is invalid, try to infer from POI name
        if is_location_invalid:
            print(f"📍 場所が不明なため、POI名から推測します: {poi_name}")
            inferred_locations = self._infer_location_from_poi(poi_name)
            
            # If we have inferred locations, use them
            if inferred_locations:
                print(f"   推測された場所: {', '.join(inferred_locations)}")
                for loc in inferred_locations:
                    variations.append(f"{poi_name}|{loc}")
            
            # Also try without location
            variations.append(poi_name)
            
            # Try with generic Japan location
            variations.append(f"{poi_name}|Japan")
            
            # Try to extract city from POI name itself
            major_cities = ["Tokyo", "Osaka", "Kyoto", "Naha", "Okinawa", "Sapporo", "Fukuoka", "Nagoya", "Yokohama", "Kobe", "Hiroshima"]
            for city in major_cities:
                if city.lower() in poi_name.lower():
                    variations.append(f"{poi_name}|{city}, Japan")
                    variations.append(f"{poi_name}|{city}")
                    break
        
        # Only process location context if it's valid
        else:
            # 1. Original search term with location
            variations.append(f"{poi_name}|{location_context}")
            
            # 2. Add common Japanese suffixes/prefixes if not present
            japanese_landmarks = ['寺', '神社', '城', '公園', '美術館', '博物館', '駅', 'タワー', 'ビル']
            has_japanese = any(landmark in poi_name for landmark in japanese_landmarks)
            
            if not has_japanese:
                # Try common tourist spot patterns
                if 'temple' in poi_name.lower():
                    variations.append(f"{poi_name.replace('Temple', '寺').replace('temple', '寺')}|{location_context}")
                if 'shrine' in poi_name.lower():
                    variations.append(f"{poi_name.replace('Shrine', '神社').replace('shrine', '神社')}|{location_context}")
                if 'castle' in poi_name.lower():
                    variations.append(f"{poi_name.replace('Castle', '城').replace('castle', '城')}|{location_context}")
                if 'tower' in poi_name.lower():
                    variations.append(f"{poi_name.replace('Tower', 'タワー').replace('tower', 'タワー')}|{location_context}")
            
            # 3. Try removing common words that might confuse search
            clean_name = poi_name.replace('The ', '').replace('the ', '')
            if clean_name != poi_name:
                variations.append(f"{clean_name}|{location_context}")
            
            # 4. Try with more specific location if available
            # Extract city from location context (e.g., "Tokyo, Japan" -> "Tokyo")
            city = location_context.split(',')[0].strip()
            if city != location_context and city.lower() not in invalid_locations:
                variations.append(f"{poi_name}|{city}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for v in variations:
            if v not in seen:
                seen.add(v)
                unique_variations.append(v)
        
        return unique_variations
    
    def _infer_location_from_poi(self, poi_name: str) -> List[str]:
        """Infer possible locations from POI name using LLM"""
        locations = []
        
        try:
            # Use LLM to infer location
            prompt = f"""
Analyze this POI (Point of Interest) name and infer its most likely location(s).
POI Name: {poi_name}

Based on the POI name, provide up to 3 possible locations where this POI might be located.
Consider:
1. If it's an airport, station, or landmark with a city/region name
2. Famous tourist attractions and their known locations
3. Cultural or linguistic clues in the name

Return ONLY a JSON array of location strings in the format ["City, Country"] or ["City"].
If you're confident about the country, include it. Otherwise, just the city.
Focus on specific cities rather than regions when possible.

Example response for "Naha Airport": ["Okinawa, Japan", "Naha, Japan", "Okinawa"]
Example response for "Eiffel Tower": ["Paris, France", "Paris"]

Response:"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()
            
            # Parse JSON response
            if response_text.startswith('['):
                try:
                    # Try to parse as JSON directly
                    import json
                    locations = json.loads(response_text)
                except:
                    # If JSON parsing fails, try to extract array
                    import re
                    match = re.search(r'\[([^\]]+)\]', response_text)
                    if match:
                        # Clean up and parse
                        items = match.group(1).split(',')
                        locations = [item.strip().strip('"').strip("'") for item in items]
            
            # If no locations found through LLM, provide generic fallback
            if not locations:
                locations = ['Japan']  # Default fallback for Japanese context
            
        except Exception as e:
            print(f"⚠️ LLM場所推測エラー: {str(e)}")
            # Fallback to simple heuristics if LLM fails
            poi_lower = poi_name.lower()
            if 'airport' in poi_lower or '空港' in poi_name:
                locations = ['Japan']
            elif 'station' in poi_lower or '駅' in poi_name:
                locations = ['Japan']
            else:
                locations = ['Japan']
        
        # Ensure we return a list
        if isinstance(locations, str):
            locations = [locations]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_locations = []
        for loc in locations:
            if loc and loc not in seen:
                seen.add(loc)
                unique_locations.append(loc)
        
        return unique_locations

class PlanAgent(BaseAgent):
    def __init__(self, llm: SupportedChatModel, tools: List):
        # Include restaurant search tool for meal planning
        relevant_tools = [tool for tool in tools if 'route' in tool.name.lower() or 'constraint' in tool.name.lower() or 'restaurant' in tool.name.lower()]
        super().__init__(llm, relevant_tools)
        self.name = "Plan Agent"
    
    def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create and optimize travel itinerary using prompt.py prompts"""
        try:
            # Clear tool logs (starting new execution)
            tool_logger.clear_logs()
            
            poi_data = context.get('poi_data', [])
            people_count = context.get('people_count', 1)
            days = context.get('days', 3)
            budget = context.get('budget', 3000)  # USD
            route_info = context.get('route_info', {})
            video_frames = context.get('video_frames', [])
            video_insights = context.get('video_insights', {})
            has_video = context.get('has_video', False)
            
            # Filter out POIs with errors
            # valid_pois = [poi for poi in poi_data if 'error' not in poi]
            valid_pois = poi_data
            
            # Use the proper prompt from prompt.py
            prompt_text = PLAN_AGENT_PROMPT.format(
                poi_data=json.dumps(valid_pois, indent=2),
                route_info=json.dumps(route_info, indent=2),
                people_count=people_count,
                days=days,
                budget=budget
            )
            
            # Add tool usage instructions and constraint evaluation
            full_prompt = f"""
            {prompt_text}
            
            {TOOL_USAGE_INSTRUCTIONS}
            
            Current Planning Task: {task}
            
            Create a detailed, constraint-aware itinerary that:
            1. Prioritizes POIs with higher importance scores
            2. Ensures ${budget} USD budget adherence for {people_count} people
            3. Fits realistically into {days} days
            4. Considers group logistics and coordination
            5. Provides clear rationale for inclusions/exclusions
            
            If budget/time constraints require POI cuts, prioritize by importance score.
            Provide detailed reasoning for all decisions.
            """
            
            # If video is available and using Gemini, analyze it for better planning
            video_file = context.get('video_file')
            if video_file and self.is_gemini:
                print("🎥 プランエージェントが動画を含めて分析中...")
                
                # Add video context to prompt
                video_enhanced_prompt = full_prompt + """
                
Additionally, analyze the provided travel video to:
1. Understand the travel style and preferences shown
2. Consider the pace and group dynamics observed
3. Incorporate insights about activities and experiences
4. Align the itinerary with the video's travel patterns
"""
                
                planning_response_text = self._analyze_with_video(video_enhanced_prompt, video_file)
            else:
                # Standard text-only analysis
                planning_response = self.llm.invoke([HumanMessage(content=full_prompt)])
                planning_response_text = planning_response.content
                self._track_token_usage(self.name, full_prompt, planning_response_text, planning_response)
            
            # Extract selected POIs with constraint considerations
            selected_pois = self._extract_selected_pois_with_constraints(
                planning_response_text, valid_pois, people_count, days, budget
            )
            
            # Search for restaurants near selected POIs
            restaurant_recommendations = self._search_restaurants_for_pois(
                selected_pois, people_count, budget
            )
            
            # Create enhanced itinerary with restaurant recommendations
            enhanced_itinerary = self._enhance_itinerary_with_restaurants(
                planning_response_text, restaurant_recommendations, selected_pois
            )
            
            result = {
                'agent': self.name,
                'task_completed': True,
                'itinerary': enhanced_itinerary,
                'selected_pois': selected_pois,
                'restaurant_recommendations': restaurant_recommendations,
                'total_available_pois': len(valid_pois),
                'selected_poi_count': len(selected_pois),
                'total_restaurant_recommendations': sum(len(r['restaurants']) for r in restaurant_recommendations.values()),
                'constraints': {
                    'people_count': people_count,
                    'days': days,
                    'budget_usd': budget
                },
                'tools_used': [tool.name for tool in self.tools],
                'tool_execution_count': len(tool_logger.execution_logs)
            }
            
            self.memory.append(result)
            return result
            
        except Exception as e:
            return {
                'agent': self.name,
                'task_completed': False,
                'error': str(e),
                'itinerary': '',
                'selected_pois': []
            }
    
    def _extract_selected_pois_with_constraints(self, itinerary: str, poi_data: List[Dict], people_count: int, days: int, budget: float) -> List[Dict]:
        """Extract selected POIs considering constraints and importance scores"""
        selected_pois = []
        itinerary_lower = itinerary.lower()
        
        # Sort POIs by importance score (higher first)
        sorted_pois = sorted(poi_data, key=lambda x: x.get('importance_score', 0), reverse=True)
        
        for poi in sorted_pois:
            if poi.get('name') and poi['name'].lower() in itinerary_lower:
                selected_pois.append(poi)
        
        # Apply constraint-based filtering if needed
        max_pois_per_day = 3  # Realistic limit
        max_total_pois = days * max_pois_per_day
        
        if len(selected_pois) > max_total_pois:
            # Keep highest importance POIs within constraint
            selected_pois = selected_pois[:max_total_pois]
        
        return selected_pois
    
    def _search_restaurants_for_pois(self, selected_pois: List[Dict], people_count: int, budget: float) -> Dict[str, Any]:
        """Search for restaurants near each selected POI"""
        restaurant_recommendations = {}
        
        # Get restaurant search tool
        restaurant_tool = next((tool for tool in self.tools if 'restaurant' in tool.name), None)
        
        if not restaurant_tool:
            print("⚠️ Restaurant search tool not found")
            return restaurant_recommendations
        
        # Calculate appropriate price level based on budget
        daily_budget_per_person = budget / (people_count * 3) if people_count > 0 else budget / 3
        # Allocate roughly 30% of daily budget for meals
        meal_budget_per_person = daily_budget_per_person * 0.3
        
        # Map budget to price level (rough estimation)
        if meal_budget_per_person < 30:
            max_price_level = 1  # Budget-friendly
        elif meal_budget_per_person < 60:
            max_price_level = 2  # Moderate
        elif meal_budget_per_person < 100:
            max_price_level = 3  # Expensive
        else:
            max_price_level = 4  # Very expensive
        
        print(f"\n🍽️ レストラン検索を開始します（予算レベル: {max_price_level}/4）")
        
        for poi in selected_pois:
            poi_name = poi.get('name', '')
            poi_address = poi.get('address', poi_name)
            
            print(f"\n📍 {poi_name} 周辺のレストランを検索中...")
            
            # Search for restaurants near this POI
            search_query = f"{poi_address}|1000|{max_price_level}"
            restaurants = restaurant_tool.func(search_query)
            
            if isinstance(restaurants, list) and len(restaurants) > 0 and 'error' not in restaurants[0]:
                # Filter out any error results
                valid_restaurants = [r for r in restaurants if 'error' not in r]
                
                # Categorize restaurants by meal type
                lunch_restaurants = []
                dinner_restaurants = []
                
                for restaurant in valid_restaurants[:6]:  # Top 6 restaurants
                    if restaurant.get('recommended_for') == 'Breakfast/Brunch':
                        # Skip breakfast places for main meal recommendations
                        continue
                    elif restaurant.get('recommended_for') == 'Dinner/Evening':
                        dinner_restaurants.append(restaurant)
                    else:
                        # Can be used for both lunch and dinner
                        if len(lunch_restaurants) < 2:
                            lunch_restaurants.append(restaurant)
                        else:
                            dinner_restaurants.append(restaurant)
                
                restaurant_recommendations[poi_name] = {
                    'poi_address': poi_address,
                    'restaurants': valid_restaurants[:3],  # Top 3 overall
                    'lunch_options': lunch_restaurants[:2],  # Top 2 for lunch
                    'dinner_options': dinner_restaurants[:2],  # Top 2 for dinner
                    'search_radius': 1000,
                    'max_price_level': max_price_level
                }
                
                print(f"   ✅ {len(valid_restaurants)} 件のレストランが見つかりました")
            else:
                print(f"   ⚠️ レストランが見つかりませんでした")
                restaurant_recommendations[poi_name] = {
                    'poi_address': poi_address,
                    'restaurants': [],
                    'lunch_options': [],
                    'dinner_options': [],
                    'search_radius': 1000,
                    'max_price_level': max_price_level,
                    'error': 'No restaurants found'
                }
        
        return restaurant_recommendations
    
    def _enhance_itinerary_with_restaurants(self, original_itinerary: str, restaurant_recommendations: Dict, selected_pois: List[Dict]) -> str:
        """Enhance the itinerary with specific restaurant recommendations"""
        # Use LLM to create an enhanced itinerary with restaurant details
        restaurant_summary = self._format_restaurant_recommendations(restaurant_recommendations)
        
        enhancement_prompt = f"""
Based on the original itinerary and restaurant search results, create an enhanced itinerary that includes specific restaurant recommendations.

Original Itinerary:
{original_itinerary}

Restaurant Recommendations Found:
{restaurant_summary}

Please enhance the itinerary by:
1. Adding specific restaurant names and addresses for lunch and dinner
2. Including cuisine types and price ranges
3. Explaining why each restaurant is recommended (ratings, proximity to POIs, cuisine match)
4. Suggesting meal timing that fits with the POI visit schedule
5. Providing ordering recommendations or specialties for each restaurant

Format the enhanced itinerary in a natural, flowing style that integrates the restaurant recommendations seamlessly into the daily plans.
"""
        
        try:
            response = self.llm.invoke([HumanMessage(content=enhancement_prompt)])
            enhanced_itinerary = response.content
            
            # Track token usage
            self._track_token_usage(self.name, enhancement_prompt, enhanced_itinerary, response)
            
            return enhanced_itinerary
        except Exception as e:
            print(f"⚠️ イティネラリー強化エラー: {str(e)}")
            # Return original with basic restaurant info appended
            return f"{original_itinerary}\n\n## Restaurant Recommendations\n{restaurant_summary}"
    
    def _format_restaurant_recommendations(self, recommendations: Dict) -> str:
        """Format restaurant recommendations for display"""
        formatted = []
        
        for poi_name, rec_data in recommendations.items():
            formatted.append(f"\n### Near {poi_name}")
            
            if rec_data.get('lunch_options'):
                formatted.append("\n**Lunch Options:**")
                for restaurant in rec_data['lunch_options']:
                    formatted.append(f"- **{restaurant['name']}**")
                    formatted.append(f"  - Address: {restaurant['address']}")
                    formatted.append(f"  - Cuisine: {restaurant['cuisine_type']}")
                    formatted.append(f"  - Price: {restaurant['price_level_text']}")
                    formatted.append(f"  - Rating: {restaurant['rating']} ⭐ ({restaurant['user_ratings_total']} reviews)")
                    if restaurant.get('editorial_summary'):
                        formatted.append(f"  - About: {restaurant['editorial_summary']}")
            
            if rec_data.get('dinner_options'):
                formatted.append("\n**Dinner Options:**")
                for restaurant in rec_data['dinner_options']:
                    formatted.append(f"- **{restaurant['name']}**")
                    formatted.append(f"  - Address: {restaurant['address']}")
                    formatted.append(f"  - Cuisine: {restaurant['cuisine_type']}")
                    formatted.append(f"  - Price: {restaurant['price_level_text']}")
                    formatted.append(f"  - Rating: {restaurant['rating']} ⭐ ({restaurant['user_ratings_total']} reviews)")
                    if restaurant.get('editorial_summary'):
                        formatted.append(f"  - About: {restaurant['editorial_summary']}")
        
        return '\n'.join(formatted)

class RouteSearchAgent(BaseAgent):
    def __init__(self, llm: SupportedChatModel, tools: List):
        # Only use route/transportation-related tools (including browser_use)
        relevant_tools = [tool for tool in tools if 'route' in tool.name.lower() or 'transport' in tool.name.lower() or 'direction' in tool.name.lower() or 'browser_use' in tool.name.lower()]
        super().__init__(llm, relevant_tools)
        self.name = "Route Search Agent"
    
    def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Find transportation options using prompt.py prompts"""
        try:
            # Clear tool logs (starting new execution)
            tool_logger.clear_logs()
            
            selected_pois = context.get('selected_pois', [])
            people_count = context.get('people_count', 1)
            budget = context.get('budget', 3000)
            transport_prefs = context.get('transport_prefs', 'public_transport')
            
            if len(selected_pois) < 2:
                return {
                    'agent': self.name,
                    'task_completed': False,
                    'error': 'Need at least 2 POIs for route planning',
                    'routes': []
                }
            
            # Create POI order for route planning
            poi_order = [poi.get('name', f"POI_{i}") for i, poi in enumerate(selected_pois)]
            
            # Use the proper prompt from prompt.py
            prompt_text = ROUTE_SEARCH_AGENT_PROMPT.format(
                poi_order=json.dumps(poi_order),
                people_count=people_count,
                budget=budget,
                transport_prefs=transport_prefs
            )
            
            full_prompt = f"""
            {prompt_text}
            
            {TOOL_USAGE_INSTRUCTIONS}
            
            Current Task: {task}
            
            Find optimal transportation between these POIs:
            {json.dumps(poi_order, indent=2)}
            
            Consider:
            - Group size: {people_count} people (coordination requirements)
            - Total budget: ${budget} USD (transportation portion)
            - Cost-effectiveness vs convenience trade-offs
            - Local transportation options and group discounts
            - Travel time optimization for group schedule
            
            Provide detailed route options with cost estimates in USD.
            """
            
            # Use route search tools if available
            route_tool = next((tool for tool in self.tools if 'route' in tool.name), None)
            routes_data = []
            
            if route_tool and len(selected_pois) >= 2:
                # Get routes between consecutive POIs
                for i in range(len(selected_pois) - 1):
                    origin_poi = selected_pois[i]
                    dest_poi = selected_pois[i + 1]
                    
                    origin = origin_poi.get('address', origin_poi.get('name', ''))
                    destination = dest_poi.get('address', dest_poi.get('name', ''))
                    
                    if origin and destination:
                        print(f"\n📍 Searching route {i+1}: {origin_poi.get('name')} -> {dest_poi.get('name')}")
                        
                        # Determine the best transport mode for this specific route
                        best_mode = self._determine_best_transport_mode(
                            origin_poi, dest_poi, people_count, budget, context.get('location_context', '')
                        )
                        
                        print(f"   🚗🚃 適切な交通手段を判断中...")
                        
                        # Try both modes and compare
                        transit_route = route_tool.func(f"{origin}|{destination}|TRANSIT")
                        drive_route = route_tool.func(f"{origin}|{destination}|DRIVE")
                        
                        # Select the best route based on analysis
                        selected_route, selected_mode = self._select_best_route(
                            transit_route, drive_route, origin_poi, dest_poi, people_count, budget
                        )
                        
                        if 'error' not in selected_route and selected_route.get('routes'):
                            routes_data.append({
                                'from': origin_poi.get('name'),
                                'to': dest_poi.get('name'),
                                'route_info': selected_route,
                                'recommended_mode': selected_mode,
                                'segment': f"{i+1}-{i+2}"
                            })
                            print(f"   ✅ 推奨交通手段: {selected_mode}")
                        else:
                            print(f"   ⚠️ No route found for this segment")
                            # Still add to data but mark as no route found
                            routes_data.append({
                                'from': origin_poi.get('name'),
                                'to': dest_poi.get('name'),
                                'route_info': {'routes': [], 'error': 'No routes found'},
                                'recommended_mode': 'unknown',
                                'segment': f"{i+1}-{i+2}"
                            })
            
            # Get LLM analysis with route data
            analysis_prompt = f"""
            {full_prompt}
            
            Route Data Found: {json.dumps(routes_data, indent=2)}
            
            Analyze transportation options and provide:
            1. Recommended transport mode for group of {people_count}
            2. Cost estimates (total and per-person) in USD
            3. Time considerations and coordination tips
            4. Alternative options and trade-offs
            5. Group-specific logistics recommendations
            """
            
            response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            
            # Track token usage
            self._track_token_usage(self.name, analysis_prompt, response.content)
            
            # Determine optimal transport mode for group
            transport_mode = self._determine_transport_mode(people_count, budget)
            group_advice = self._get_group_transport_advice(people_count, budget)
            
            result = {
                'agent': self.name,
                'task_completed': True,
                'routes': routes_data,
                'transport_analysis': response.content,
                'recommended_mode': transport_mode,
                'group_advice': group_advice,
                'poi_sequence': poi_order,
                'total_segments': len(routes_data),
                'tools_used': [tool.name for tool in self.tools],
                'tool_execution_count': len(tool_logger.execution_logs)
            }
            
            self.memory.append(result)
            return result
            
        except Exception as e:
            return {
                'agent': self.name,
                'task_completed': False,
                'error': str(e),
                'routes': []
            }
    
    def _determine_transport_mode(self, people_count: int, budget: float) -> str:
        """Determine optimal transport mode based on group size and budget"""
        if people_count <= 2:
            return "public_transport" if budget < 1000 else "taxi_rideshare"
        elif people_count <= 4:
            return "rideshare_multiple" if budget < 2000 else "private_car_rental"
        else:
            return "group_transport" if budget < 3000 else "private_van_rental"
    
    def _get_group_transport_advice(self, people_count: int, budget: float) -> str:
        """Get specific advice for group transportation"""
        advice = []
        
        if people_count >= 4:
            advice.append("Consider splitting into smaller groups for flexibility")
            advice.append("Book transportation in advance for group discounts")
        
        if budget > 2000:
            advice.append("Private transportation may be cost-effective for groups")
        else:
            advice.append("Public transport with group tickets recommended")
        
        return "; ".join(advice)
    
    def _determine_best_transport_mode(self, origin_poi: Dict, dest_poi: Dict, 
                                     people_count: int, budget: float, location_context: str) -> str:
        """Intelligently determine the best transport mode for a specific route"""
        try:
            # Use LLM to analyze the route
            prompt = f"""
Analyze this route and determine the most appropriate transportation mode:

From: {origin_poi.get('name')} ({origin_poi.get('address', 'Unknown address')})
To: {dest_poi.get('name')} ({dest_poi.get('address', 'Unknown address')})
Location: {location_context}
Group size: {people_count} people
Budget: ${budget} USD

Consider:
1. Distance between locations
2. Availability of public transportation in the area
3. POI types (e.g., airport usually needs car/taxi, city centers have good transit)
4. Group size and luggage considerations
5. Cost efficiency

Return ONLY one of: "TRANSIT" or "DRIVE"
If unsure, prefer "TRANSIT" for city areas and "DRIVE" for remote locations.
"""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            mode = response.content.strip().upper()
            
            # Validate response
            if mode not in ["TRANSIT", "DRIVE"]:
                # Fallback logic based on POI types
                origin_name = origin_poi.get('name', '').lower()
                dest_name = dest_poi.get('name', '').lower()
                
                # Airport routes usually need car/taxi
                if 'airport' in origin_name or 'airport' in dest_name:
                    return "DRIVE"
                # Hotels and remote resorts often need car
                elif 'resort' in origin_name or 'resort' in dest_name:
                    return "DRIVE"
                # Default to transit for city locations
                else:
                    return "TRANSIT"
            
            return mode
            
        except Exception as e:
            print(f"⚠️ 交通手段判断エラー: {str(e)}")
            # Default fallback
            return "TRANSIT"
    
    def _select_best_route(self, transit_route: Dict, drive_route: Dict, 
                          origin_poi: Dict, dest_poi: Dict, people_count: int, budget: float) -> tuple:
        """Compare transit and drive routes and select the best option"""
        
        # Extract route information
        transit_valid = 'error' not in transit_route and transit_route.get('routes')
        drive_valid = 'error' not in drive_route and drive_route.get('routes')
        
        # If only one is valid, return it
        if transit_valid and not drive_valid:
            return transit_route, "電車/バス"
        elif drive_valid and not transit_valid:
            return drive_route, "車/タクシー"
        elif not transit_valid and not drive_valid:
            return {'routes': [], 'error': 'No routes found'}, "unknown"
        
        # Both are valid, compare them
        try:
            # Get duration and distance for comparison
            transit_duration = transit_route['routes'][0].get('duration_value', float('inf'))
            drive_duration = drive_route['routes'][0].get('duration_value', float('inf'))
            
            # Use LLM for intelligent decision
            prompt = f"""
Compare these two route options and choose the best one:

Route: {origin_poi.get('name')} → {dest_poi.get('name')}
Group size: {people_count} people

Option 1 - Public Transit:
- Duration: {transit_route['routes'][0].get('duration', 'Unknown')}
- Routes available: {len(transit_route.get('routes', []))}

Option 2 - Car/Taxi:
- Duration: {drive_route['routes'][0].get('duration', 'Unknown')}
- Routes available: {len(drive_route.get('routes', []))}

Consider:
1. Time efficiency (significant difference?)
2. Cost for {people_count} people
3. Convenience and comfort
4. Local transportation norms

Return ONLY: "TRANSIT" or "DRIVE"
"""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            decision = response.content.strip().upper()
            
            if decision == "DRIVE":
                return drive_route, "車/タクシー"
            else:
                return transit_route, "電車/バス"
                
        except Exception as e:
            print(f"⚠️ ルート比較エラー: {str(e)}")
            # Default to transit if comparison fails
            return transit_route, "電車/バス"

class AccommodationAgent(BaseAgent):
    def __init__(self, llm: SupportedChatModel, tools: List):
        # Only use accommodation-related tools
        relevant_tools = [tool for tool in tools if 'accommodation' in tool.name.lower() or 'hotel' in tool.name.lower() or 'lodging' in tool.name.lower()]
        super().__init__(llm, relevant_tools)
        self.name = "Accommodation Agent"
    
    def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Find accommodation options using prompt.py prompts"""
        try:
            # Clear tool logs (starting new execution)
            tool_logger.clear_logs()
            
            travel_plan = context.get('itinerary', '')
            location = context.get('location_context', '')
            people_count = context.get('people_count', 1)
            days = context.get('days', 3)
            budget = context.get('budget', 3000)
            preferences = context.get('preferences', {})
            selected_pois = context.get('selected_pois', [])
            
            # Extract actual destination from itinerary if location seems incorrect
            actual_location = self._extract_actual_location(travel_plan, location, selected_pois)
            if actual_location != location:
                print(f"📍 実際の宿泊地を検出: {actual_location} (元の場所: {location})")
                location = actual_location
            
            # Validate that we have a reasonable location
            if not location or len(location) > 50 or location.lower().startswith('based on'):
                # Emergency fallback - try to get from first POI
                if selected_pois and len(selected_pois) > 0:
                    first_poi_address = selected_pois[0].get('address', '')
                    if first_poi_address:
                        fallback_city = self._extract_city_from_address(first_poi_address)
                        if fallback_city:
                            print(f"⚠️ 場所検出に失敗。POIの住所から推測: {fallback_city}")
                            location = fallback_city
                        else:
                            print("⚠️ 場所検出に失敗。デフォルトを使用: Tokyo")
                            location = "Tokyo"
                else:
                    print("⚠️ 場所検出に失敗。デフォルトを使用: Tokyo")
                    location = "Tokyo"
            
            # Calculate accommodation budget allocation
            accommodation_budget = budget * 0.4  # 40% of total budget
            cost_per_night = accommodation_budget / days if days > 0 else accommodation_budget
            
            # Use the proper prompt from prompt.py
            prompt_text = ACCOMMODATION_AGENT_PROMPT.format(
                travel_plan=travel_plan,
                location=location,
                people_count=people_count,
                days=days,
                budget=budget,
                preferences=json.dumps(preferences)
            )
            
            full_prompt = f"""
            {prompt_text}
            
            {TOOL_USAGE_INSTRUCTIONS}
            
            Current Task: {task}
            
            Find accommodation for:
            - Group size: {people_count} people
            - Duration: {days} nights
            - Budget allocation for accommodation: ${accommodation_budget:.0f} USD (40% of total budget)
            - Per night budget: ${cost_per_night:.0f} USD
            - Actual destination: {location}
            
            Consider:
            - Room configurations for {people_count} people
            - Location convenience for planned POIs
            - Group-friendly amenities
            - Budget optimization strategies
            
            Important: Search for hotels in the ACTUAL destination city where the group will stay, 
            not just the general region. Focus on practical location near transportation hubs or POIs.
            """
            
            # Use accommodation search tools if available
            accommodation_tool = next((tool for tool in self.tools if 'accommodation' in tool.name), None)
            accommodation_data = []
            search_location = location
            
            if accommodation_tool:
                # First try with the specific location
                print(f"🏨 {location}で宿泊施設を検索中...")
                accommodation_results = accommodation_tool.func(location)
                
                # Check if we got valid results
                if isinstance(accommodation_results, list) and len(accommodation_results) > 0:
                    accommodation_data = accommodation_results
                elif isinstance(accommodation_results, dict) and 'error' in accommodation_results:
                    print(f"⚠️ 検索エラー: {accommodation_results['error']}")
                    # Try with more specific location if available
                    if selected_pois and len(selected_pois) > 0:
                        first_poi_address = selected_pois[0].get('address', '')
                        if first_poi_address:
                            city_search = self._extract_city_from_address(first_poi_address)
                            if city_search and city_search != location:
                                print(f"🔄 {city_search}で再検索中...")
                                accommodation_results = accommodation_tool.func(city_search)
                                if isinstance(accommodation_results, list) and len(accommodation_results) > 0:
                                    accommodation_data = accommodation_results
                                    search_location = city_search
            
            # Get LLM analysis with specific recommendation
            analysis_prompt = f"""
            {full_prompt}
            
            Available Accommodation Data from Search Results:
            {json.dumps(accommodation_data[:5], indent=2) if accommodation_data else "No data available"}
            
            Travel Itinerary: {travel_plan[:1000]}...
            
            IMPORTANT: You MUST select ONE SPECIFIC hotel from the search results above if available.
            
            Based on the budget and location, recommend ONE SPECIFIC hotel that:
            1. Fits within the per-night budget of ${cost_per_night:.0f} USD
            2. Is conveniently located for the planned activities
            3. Offers appropriate room configuration for {people_count} people
            4. Provides good value for money
            
            If search results are available, you MUST choose from them. Look at:
            - Rating (4.0+ preferred)
            - Number of reviews (more reviews = more reliable)
            - Location convenience
            - Price level compatibility
            
            Structure your response EXACTLY as follows:
            
            ## Recommended Hotel
            **Hotel Name:** [Use EXACT name from search results if available]
            **Location:** [Use EXACT address from search results if available]
            **Price Range:** [Based on price_level_text or estimate: $ = $50-100, $$ = $100-200, $$$ = $200-300 per night]
            **Room Type:** [Recommended room configuration]
            
            ## Why This Hotel
            - [Reason 1 - mention specific rating if from search results]
            - [Reason 2 - mention location benefits]
            - [Reason 3 - mention amenities or features]
            - [Reason 4 - mention value proposition]
            
            ## Hotel Details (if from search results)
            - Rating: [X.X stars (Y reviews)]
            - Phone: [Include if available]
            - Website: [Include if available]
            
            ## Budget Analysis
            - Total accommodation budget: ${accommodation_budget:.0f} USD
            - Per night budget: ${cost_per_night:.0f} USD
            - Estimated total cost: [Calculate for {days} nights]
            - Budget status: [Within budget / Slightly over / Well under]
            
            ## Booking Tips
            - [Tip 1 - mention direct booking vs OTA]
            - [Tip 2 - mention best time to book]
            - [Tip 3 - mention cancellation policy importance]
            """
            
            response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            
            # Track token usage
            self._track_token_usage(self.name, analysis_prompt, response.content)
            
            # Extract recommended hotel from response
            recommended_hotel = self._extract_recommended_hotel(response.content, accommodation_data)
            
            result = {
                'agent': self.name,
                'task_completed': True,
                'recommended_hotel': recommended_hotel,
                'accommodation_search_results': accommodation_data[:5],  # Top 5 for reference
                'accommodation_analysis': response.content,
                'search_location': search_location,
                'budget_allocation': {
                    'total_accommodation_budget': accommodation_budget,
                    'per_night_budget': cost_per_night,
                    'per_person_per_night': cost_per_night / people_count if people_count > 0 else cost_per_night
                },
                'group_requirements': {
                    'people_count': people_count,
                    'nights': days,
                    'room_configuration_needed': self._get_room_configuration(people_count)
                },
                'tools_used': [tool.name for tool in self.tools],
                'tool_execution_count': len(tool_logger.execution_logs)
            }
            
            self.memory.append(result)
            return result
            
        except Exception as e:
            return {
                'agent': self.name,
                'task_completed': False,
                'error': str(e),
                'recommended_hotel': None,
                'accommodation_search_results': [],
                'tool_execution_count': len(tool_logger.execution_logs)
            }
    
    def _get_room_configuration(self, people_count: int) -> str:
        """Get recommended room configuration for group"""
        if people_count <= 2:
            return "1 double room or 2 single rooms"
        elif people_count <= 4:
            return "2 double rooms or 1 family room (4 beds)"
        elif people_count <= 6:
            return "3 double rooms or 2 family rooms"
        else:
            return f"Multiple rooms needed for {people_count} people - consider group booking"
    
    def _extract_actual_location(self, itinerary: str, location_context: str, selected_pois: List[Dict]) -> str:
        """Extract the actual city where accommodation is needed from the itinerary"""
        try:
            # First check if we can get location from selected POIs
            if selected_pois and len(selected_pois) > 0:
                # Try to extract city from POI addresses
                for poi in selected_pois[:3]:  # Check first 3 POIs
                    address = poi.get('address', '')
                    if address:
                        city = self._extract_city_from_address(address)
                        if city and len(city) < 30:
                            print(f"📍 POIの住所から都市を検出: {city}")
                            return city
            
            # If location_context looks like a specific city (not prefecture/region), use it
            if location_context and not any(suffix in location_context for suffix in ['県', '府', 'Prefecture', 'Region']):
                if len(location_context) < 30:
                    return location_context
            
            # Use LLM to analyze the itinerary and determine actual accommodation location
            prompt = f"""
Analyze this travel itinerary and determine the ACTUAL CITY where the travelers will need accommodation.

Current location context: {location_context}
Selected POIs: {[poi.get('name', '') for poi in selected_pois[:5]]}

Travel Itinerary (first 1500 chars):
{itinerary[:1500]}

Based on the itinerary, identify the main city where the group will stay overnight.
Return ONLY the city name (one word or short phrase), nothing else.

Examples of good responses:
- Kanazawa
- Tokyo
- Kyoto
- Hiroshima

Do NOT include:
- Explanations
- Prefecture names
- Multiple cities
- Full sentences

City:"""
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            actual_city = response.content.strip()
            
            # Clean up the response - get only the first line and remove any extra text
            actual_city = actual_city.split('\n')[0].strip()
            actual_city = actual_city.split(',')[0].strip()
            actual_city = actual_city.split('.')[0].strip()
            
            # Remove common prefixes/suffixes that might have been included
            for prefix in ['City:', 'city:', 'The city is', 'Based on']:
                if actual_city.startswith(prefix):
                    actual_city = actual_city[len(prefix):].strip()
            
            # Validate the response
            if actual_city and len(actual_city) < 30 and ' ' not in actual_city.strip():
                return actual_city
            
        except Exception as e:
            print(f"⚠️ 場所抽出エラー: {str(e)}")
        
        # If all else fails, try to clean up the location_context
        if location_context:
            # Remove prefecture suffixes
            cleaned = location_context.replace('県', '').replace('府', '').replace('Prefecture', '').strip()
            if cleaned and len(cleaned) < 30:
                return cleaned
        
        # Final fallback
        return location_context if location_context else "Tokyo"
    
    def _extract_city_from_address(self, address: str) -> str:
        """Extract city name from a full address"""
        import re
        
        if not address:
            return ""
        
        # For Japanese addresses with 市 (city marker)
        city_match = re.search(r'([^、,\s]+市)', address)
        if city_match:
            city = city_match.group(1)
            # Remove any prefecture prefix if attached
            city = city.split('県')[-1].split('府')[-1]
            return city
        
        # For addresses with 区 (ward - typically in major cities)
        ward_match = re.search(r'([^、,\s]+区)', address)
        if ward_match:
            # For wards, we want the city name, not the ward
            # Common pattern: "県○○市△△区" or just "○○市△△区"
            if '市' in address:
                city_match = re.search(r'([^、,\s]+市)', address)
                if city_match:
                    return city_match.group(1).split('県')[-1].split('府')[-1]
            # If no city marker, return major city names for known wards
            ward = ward_match.group(1)
            if any(tokyo_ward in ward for tokyo_ward in ['渋谷', '新宿', '中央', '千代田', '港']):
                return 'Tokyo'
            elif any(osaka_ward in ward for osaka_ward in ['北区', '中央区', '西区', '浪速区']):
                return 'Osaka'
        
        # For English/romanized addresses
        parts = address.split(',')
        if len(parts) >= 2:
            for i in range(len(parts) - 1, -1, -1):
                part = parts[i].strip()
                # Skip country names
                if part.lower() in ['japan', '日本', 'jp']:
                    continue
                # Skip postal codes
                if re.match(r'^\d{3}-?\d{4}$', part):
                    continue
                # Skip prefecture names
                if any(pref in part for pref in ['Prefecture', '県', '府', '-ken', '-fu']):
                    continue
                # If it's not too long and doesn't have numbers (except at end), it might be a city
                if len(part) < 30 and not re.search(r'\d', part.split('-')[0]):
                    return part.split('-')[0].strip()
        
        return ""
    
    def _extract_recommended_hotel(self, analysis: str, accommodation_data: List[Dict]) -> Dict[str, Any]:
        """Extract the recommended hotel information from the analysis"""
        recommended = {
            'name': None,
            'location': None,
            'price_range': None,
            'room_type': None,
            'from_search_results': False,
            'address': None,
            'phone': None,
            'website': None,
            'rating': None,
            'user_ratings_total': None,
            'place_id': None
        }
        
        try:
            # Extract hotel name using pattern matching
            import re
            
            # Look for hotel name pattern
            name_match = re.search(r'\*\*Hotel Name:\*\*\s*(.+?)(?:\n|\*\*)', analysis, re.IGNORECASE)
            if name_match:
                recommended['name'] = name_match.group(1).strip()
            
            # Look for location
            location_match = re.search(r'\*\*Location:\*\*\s*(.+?)(?:\n|\*\*)', analysis, re.IGNORECASE)
            if location_match:
                recommended['location'] = location_match.group(1).strip()
            
            # Look for price range
            price_match = re.search(r'\*\*Price Range:\*\*\s*(.+?)(?:\n|\*\*)', analysis, re.IGNORECASE)
            if price_match:
                recommended['price_range'] = price_match.group(1).strip()
            
            # Look for room type
            room_match = re.search(r'\*\*Room Type:\*\*\s*(.+?)(?:\n|\*\*)', analysis, re.IGNORECASE)
            if room_match:
                recommended['room_type'] = room_match.group(1).strip()
            
            # Check if the recommended hotel is from our search results
            if recommended['name'] and accommodation_data:
                for hotel in accommodation_data:
                    if hotel.get('name') and recommended['name'].lower() in hotel['name'].lower():
                        recommended['from_search_results'] = True
                        # Enhance with all available actual data
                        recommended['address'] = hotel.get('address')
                        recommended['phone'] = hotel.get('phone')
                        recommended['website'] = hotel.get('website')
                        recommended['rating'] = hotel.get('rating')
                        recommended['user_ratings_total'] = hotel.get('user_ratings_total')
                        recommended['place_id'] = hotel.get('place_id')
                        recommended['price_level_text'] = hotel.get('price_level_text', '')
                        
                        # If location not extracted from analysis, use address
                        if not recommended['location'] and hotel.get('address'):
                            recommended['location'] = hotel['address']
                        break
            
        except Exception as e:
            print(f"⚠️ ホテル情報抽出エラー: {str(e)}")
        
        return recommended

class SummaryAgent(BaseAgent):
    def __init__(self, llm: SupportedChatModel, tools: List):
        # Summary agent might need all tools for comprehensive analysis
        super().__init__(llm, tools)
        self.name = "Summary Agent"
    
    def _format_hotel_recommendation(self, hotel: Dict[str, Any]) -> str:
        """Format hotel recommendation with all available details"""
        if not hotel or not hotel.get('name'):
            return "No specific hotel recommended"
        
        details = [f"**{hotel['name']}**"]
        
        if hotel.get('address'):
            details.append(f"📍 Address: {hotel['address']}")
        elif hotel.get('location'):
            details.append(f"📍 Location: {hotel['location']}")
        
        if hotel.get('rating'):
            reviews = f" ({hotel.get('user_ratings_total', 0)} reviews)" if hotel.get('user_ratings_total') else ""
            details.append(f"⭐ Rating: {hotel['rating']} stars{reviews}")
        
        if hotel.get('price_range'):
            details.append(f"💰 Price: {hotel['price_range']}")
        
        if hotel.get('phone'):
            details.append(f"📞 Phone: {hotel['phone']}")
        
        if hotel.get('website'):
            details.append(f"🌐 Website: {hotel['website']}")
        
        if hotel.get('room_type'):
            details.append(f"🛏️ Recommended room: {hotel['room_type']}")
        
        return "\n".join(details)
    
    def execute(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create comprehensive travel summary using prompt.py prompts"""
        try:
            poi_data = context.get('poi_data', [])
            itinerary = context.get('itinerary', '')
            routes = context.get('routes', [])
            accommodation = context.get('accommodation_options', [])
            recommended_hotel = context.get('recommended_hotel', None)
            restaurant_recommendations = context.get('restaurant_recommendations', {})
            people_count = context.get('people_count', 1)
            days = context.get('days', 3)
            budget = context.get('budget', 3000)
            video_frames = context.get('video_frames', [])
            video_insights = context.get('video_insights', {})
            has_video = context.get('has_video', False)
            
            # Use the proper prompt from prompt.py
            prompt_text = SUMMARY_AGENT_PROMPT.format(
                poi_data=json.dumps(poi_data, indent=2),
                itinerary=itinerary,
                routes=json.dumps(routes, indent=2),
                accommodation=json.dumps(accommodation[:3], indent=2),  # Top 3 accommodations
                people_count=people_count,
                days=days,
                budget=budget
            )
            
            # Format restaurant recommendations for the prompt
            restaurant_summary = ""
            if restaurant_recommendations:
                restaurant_summary = "\n\n## Restaurant Recommendations Found by Plan Agent:\n"
                for poi_name, rec_data in restaurant_recommendations.items():
                    restaurant_summary += f"\n### Near {poi_name}:\n"
                    if rec_data.get('lunch_options'):
                        restaurant_summary += "**Lunch Options:**\n"
                        for r in rec_data['lunch_options']:
                            restaurant_summary += f"- {r['name']} ({r['cuisine_type']}, {r['price_level_text']})\n"
                    if rec_data.get('dinner_options'):
                        restaurant_summary += "**Dinner Options:**\n"
                        for r in rec_data['dinner_options']:
                            restaurant_summary += f"- {r['name']} ({r['cuisine_type']}, {r['price_level_text']})\n"
            
            full_prompt = f"""
            {prompt_text}
            
            {TOOL_USAGE_INSTRUCTIONS}
            
            Current Task: {task}
            
            {restaurant_summary}
            
            Create a comprehensive, traveler-friendly guide in natural English that reads like 
            a professional travel itinerary. Focus on the travel experience itself, not technical details.
            
            IMPORTANT: Include the specific restaurant names and details provided above in your day-by-day itinerary.
            Do NOT use generic phrases like "enjoy lunch at local restaurants" - use the actual restaurant names.
            
            Structure your guide as follows:
            
            ## Trip Overview
            - Destination highlights and what makes this trip special
            - Best time to visit and weather considerations
            - Group dynamics for {people_count} people
            
            ## Day-by-Day Itinerary
            - Morning, afternoon, and evening activities
            - Recommended visit durations for each POI
            - Transportation between locations
            - SPECIFIC meal recommendations with restaurant names, addresses, and why they're chosen
            
            ## Accommodation
            {self._format_hotel_recommendation(recommended_hotel) if recommended_hotel else "No specific hotel recommended"}
            - Why this accommodation works for your group
            - Room configuration options
            - Nearby amenities and transportation
            
            ## Budget Breakdown
            - Per-person and total group costs
            - Daily spending recommendations
            - Money-saving tips
            
            ## Practical Travel Tips
            - What to pack
            - Local customs and etiquette
            - Group coordination strategies
            - Emergency contacts and contingencies
            
            Make the guide engaging and inspiring, focusing on the experiences and memories 
            the travelers will create. Write as if you're a knowledgeable friend sharing advice.
            """
            
            # Generate budget analysis using dedicated prompt
            budget_analysis_prompt = BUDGET_ANALYSIS_PROMPT.format(
                total_budget=budget,
                people_count=people_count,
                days=days,
                selected_pois=json.dumps([poi.get('name', '') for poi in poi_data]),
                transportation=json.dumps(routes, indent=2),
                accommodation=json.dumps(accommodation[:1], indent=2)
            )
            
            # If video is available and using Gemini, create visual summary
            video_file = context.get('video_file')
            if video_file and self.is_gemini:
                print("🎥 サマリーエージェントが動画全体を参照して総合プランを作成中...")
                
                # Add video context to prompt
                video_enhanced_prompt = full_prompt + f"""
                
TRAVEL VIDEO REFERENCE:
You have access to the complete travel video. The plan is based on the actual travel experiences shown.

Please review the entire video and in your summary, include:
1. How specific moments in the video influenced the recommendations
2. Key scenes or experiences that support the choices
3. The overall atmosphere and vibe captured in the video
4. Practical tips and warnings observed from the actual footage
5. The journey flow and transitions shown in the video
6. Any hidden gems or unexpected discoveries featured

Make the summary feel like it's based on real experience, not just theoretical planning.
"""
                
                summary_response_text = self._analyze_with_video(video_enhanced_prompt, video_file)
            else:
                # Standard text-only analysis
                summary_response = self.llm.invoke([HumanMessage(content=full_prompt)])
                summary_response_text = summary_response.content
                self._track_token_usage(self.name, full_prompt, summary_response_text, summary_response)
            
            # Get detailed budget analysis
            budget_response = self.llm.invoke([HumanMessage(content=budget_analysis_prompt)])
            
            # Track token usage for budget analysis
            self._track_token_usage(self.name, budget_analysis_prompt, budget_response.content, budget_response)
            
            result = {
                'agent': self.name,
                'task_completed': True,
                'comprehensive_summary': summary_response_text,
                'detailed_budget_analysis': budget_response.content,
                'summary_components': {
                    'pois_included': len([poi for poi in poi_data if 'error' not in poi]),
                    'transport_segments': len(routes),
                    'accommodation_options': len(accommodation),
                    'total_budget_usd': budget,
                    'group_size': people_count,
                    'trip_duration': days
                },
                'tools_used': [tool.name for tool in self.tools],
                'tool_execution_count': len(tool_logger.execution_logs)
            }
            
            self.memory.append(result)
            return result
            
        except Exception as e:
            return {
                'agent': self.name,
                'task_completed': False,
                'error': str(e),
                'comprehensive_summary': '',
                'detailed_budget_analysis': '',
                'tool_execution_count': len(tool_logger.execution_logs)
            }

def create_agents(llm: SupportedChatModel) -> Dict[str, BaseAgent]:
    """Create all agents with appropriate tools for each"""
    # Get all tools
    all_tools = create_tools()
    
    # Create agents with tool filtering
    agents = {
        'google_map_agent': GoogleMapAgent(llm, all_tools),
        'plan_agent': PlanAgent(llm, all_tools),
        'route_search_agent': RouteSearchAgent(llm, all_tools),
        'accommodation_agent': AccommodationAgent(llm, all_tools),
        'summary_agent': SummaryAgent(llm, all_tools)
    }
    
    return agents 