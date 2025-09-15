import os
import requests
import datetime
import time
import json
from typing import List, Dict, Any
from langchain.tools import Tool
from dotenv import load_dotenv

load_dotenv()

class ToolExecutionLogger:
    """Class to record detailed tool execution logs"""
    
    def __init__(self):
        self.execution_logs = []
    
    def log_execution(self, tool_name: str, input_data: Any, output_data: Any, 
                     execution_time: float, success: bool = True, error: str = None, 
                     api_details: Dict = None):
        """Record tool execution log"""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "tool_name": tool_name,
            "input": input_data,
            "output": output_data,
            "execution_time_seconds": execution_time,
            "success": success,
            "error": error,
            "api_details": api_details or {}
        }
        self.execution_logs.append(log_entry)
        
        # Also output to console
        status = "✅" if success else "❌"
        print(f"{status} {tool_name}: {execution_time:.2f}s - Input: {str(input_data)[:100]}...")
        if error:
            print(f"   Error: {error}")
    
    def get_logs(self) -> List[Dict]:
        """Get all logs"""
        return self.execution_logs
    
    def clear_logs(self):
        """Clear logs"""
        self.execution_logs = []

# Global logger instance
tool_logger = ToolExecutionLogger()

class GoogleMapsAPITool:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        self.places_base_url = "https://places.googleapis.com/v1"
        self.routes_base_url = "https://routes.googleapis.com/v1"
    
    def get_poi_details(self, poi_name: str, location_context: str = "") -> Dict:
        """Get detailed information for a specific POI using Places API (New)"""
        start_time = time.time()
        input_data = {"poi_name": poi_name, "location_context": location_context}
        
        try:
            # First, search for the POI using Text Search
            search_query = f"{poi_name} {location_context}".strip()
            search_result = self._text_search(search_query)
            
            if not search_result.get('places'):
                error_msg = f"POI '{poi_name}' not found"
                execution_time = time.time() - start_time
                output_data = {"error": error_msg}
                tool_logger.log_execution(
                    "poi_details_search", input_data, output_data, 
                    execution_time, False, error_msg
                )
                return output_data
            
            # Get the first (most relevant) result
            place = search_result['places'][0]
            place_id = place.get('id')
            
            if not place_id:
                error_msg = f"No place ID found for '{poi_name}'"
                execution_time = time.time() - start_time
                output_data = {"error": error_msg}
                tool_logger.log_execution(
                    "poi_details_search", input_data, output_data, 
                    execution_time, False, error_msg
                )
                return output_data
            
            # Get detailed information using Place Details
            poi_details = self._get_place_details(place_id)
            
            if 'error' in poi_details:
                execution_time = time.time() - start_time
                tool_logger.log_execution(
                    "poi_details_search", input_data, poi_details, 
                    execution_time, False, poi_details['error']
                )
                return poi_details
            
            poi_info = {
                'name': poi_details.get('displayName', {}).get('text', poi_name),
                'address': poi_details.get('formattedAddress', ''),
                'rating': poi_details.get('rating', 0),
                'user_ratings_total': poi_details.get('userRatingCount', 0),
                'price_level': self._extract_price_level(poi_details),
                'types': poi_details.get('types', []),
                'place_id': place_id,
                'location': poi_details.get('location', {}),
                'opening_hours': self._extract_opening_hours(poi_details),
                'website': poi_details.get('websiteUri', ''),
                'phone': poi_details.get('nationalPhoneNumber', ''),
                'reviews': poi_details.get('reviews', [])[:3],  # Top 3 reviews
                'photos_count': len(poi_details.get('photos', [])),
                'importance_score': self._calculate_importance_score(poi_details)
            }
            
            execution_time = time.time() - start_time
            api_details = {
                "search_query": search_query,
                "place_id": place_id,
                "raw_search_results_count": len(search_result.get('places', [])),
                "api_calls": ["text_search", "place_details"]
            }
            
            tool_logger.log_execution(
                "poi_details_search", input_data, poi_info, 
                execution_time, True, None, api_details
            )
            
            return poi_info
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Failed to get POI details for '{poi_name}': {str(e)}"
            output_data = {"error": error_msg}
            tool_logger.log_execution(
                "poi_details_search", input_data, output_data, 
                execution_time, False, error_msg
            )
            return output_data
    
    def _text_search(self, query: str) -> Dict:
        """Perform text search using Places API (New)"""
        url = f"{self.places_base_url}/places:searchText"
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': self.api_key,
            'X-Goog-FieldMask': 'places.id,places.displayName,places.formattedAddress,places.rating,places.types'
        }
        
        data = {
            "textQuery": query,
            "maxResultCount": 5,
            "includedType": "tourist_attraction"
        }
        
        try:
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Text search failed: {response.status_code} - {response.text}"}
        except Exception as e:
            return {"error": f"Text search request failed: {str(e)}"}
    
    def _get_place_details(self, place_id: str) -> Dict:
        """Get detailed place information using Place Details API"""
        url = f"{self.places_base_url}/places/{place_id}"
        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': self.api_key,
            'X-Goog-FieldMask': 'id,displayName,formattedAddress,rating,userRatingCount,priceLevel,types,location,currentOpeningHours,websiteUri,nationalPhoneNumber,reviews,photos'
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Place details failed: {response.status_code} - {response.text}"}
        except Exception as e:
            return {"error": f"Place details request failed: {str(e)}"}
    
    def _extract_price_level(self, place_data: Dict) -> int:
        """Extract price level from place data"""
        price_level = place_data.get('priceLevel', 'PRICE_LEVEL_UNSPECIFIED')
        price_mapping = {
            'PRICE_LEVEL_FREE': 0,
            'PRICE_LEVEL_INEXPENSIVE': 1,
            'PRICE_LEVEL_MODERATE': 2,
            'PRICE_LEVEL_EXPENSIVE': 3,
            'PRICE_LEVEL_VERY_EXPENSIVE': 4,
            'PRICE_LEVEL_UNSPECIFIED': 0
        }
        return price_mapping.get(price_level, 0)
    
    def _extract_opening_hours(self, place_data: Dict) -> List[str]:
        """Extract opening hours from place data"""
        opening_hours = place_data.get('currentOpeningHours', {})
        weekday_descriptions = opening_hours.get('weekdayDescriptions', [])
        return weekday_descriptions
    
    def _calculate_importance_score(self, place_data: Dict) -> float:
        """Calculate importance score based on Google Maps data"""
        score = 0.0
        
        # Rating contribution (0-40 points)
        rating = place_data.get('rating', 0)
        if rating > 0:
            score += rating * 8  # Max 40 points for 5-star rating
        
        # Number of reviews contribution (0-30 points)
        reviews_count = place_data.get('userRatingCount', 0)
        if reviews_count > 0:
            # Logarithmic scale for reviews
            import math
            score += min(30, math.log10(reviews_count + 1) * 10)
        
        # Type-based importance (0-20 points)
        types = place_data.get('types', [])
        important_types = {
            'tourist_attraction': 15,
            'museum': 12,
            'amusement_park': 10,
            'zoo': 8,
            'park': 6,
            'shopping_mall': 5,
            'restaurant': 3
        }
        
        type_score = 0
        for poi_type in types:
            if poi_type in important_types:
                type_score = max(type_score, important_types[poi_type])
        score += type_score
        
        # Photos availability (0-10 points)
        photos_count = len(place_data.get('photos', []))
        score += min(10, photos_count * 2)
        
        return min(100, score)  # Cap at 100
    
    def search_restaurants(self, location: str, radius: int = 1000, price_level_max: int = 3) -> List[Dict]:
        """Search for restaurants near a location using Places API (New)"""
        start_time = time.time()
        input_data = {"location": location, "radius": radius, "price_level_max": price_level_max}
        
        try:
            url = f"{self.places_base_url}/places:searchNearby"
            headers = {
                'Content-Type': 'application/json',
                'X-Goog-Api-Key': self.api_key,
                'X-Goog-FieldMask': 'places.id,places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.priceLevel,places.types,places.location,places.editorialSummary,places.currentOpeningHours'
            }
            
            # Try to geocode the location first
            geocode_result = self._geocode_location(location)
            if 'error' in geocode_result:
                # If geocoding fails, try text search instead
                return self._search_restaurants_by_text(location, price_level_max)
            
            lat_lng = geocode_result['location']
            
            data = {
                "includedTypes": ["restaurant"],
                "excludedTypes": ["fast_food"],
                "maxResultCount": 10,
                "locationRestriction": {
                    "circle": {
                        "center": {
                            "latitude": lat_lng['lat'],
                            "longitude": lat_lng['lng']
                        },
                        "radius": radius
                    }
                },
                "rankPreference": "POPULARITY"
            }
            
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                restaurants = []
                
                for place in result.get('places', []):
                    # Filter by price level if specified
                    place_price_level = self._extract_price_level(place)
                    if place_price_level > price_level_max:
                        continue
                    
                    restaurant_info = {
                        'name': place.get('displayName', {}).get('text', 'Unknown'),
                        'address': place.get('formattedAddress', ''),
                        'rating': place.get('rating', 0),
                        'user_ratings_total': place.get('userRatingCount', 0),
                        'price_level': place_price_level,
                        'price_level_text': self._get_price_level_text(place_price_level),
                        'types': place.get('types', []),
                        'place_id': place.get('id', ''),
                        'location': place.get('location', {}),
                        'editorial_summary': place.get('editorialSummary', {}).get('text', ''),
                        'opening_hours': self._extract_opening_hours(place),
                        'cuisine_type': self._extract_cuisine_type(place),
                        'recommended_for': self._get_meal_recommendation(place)
                    }
                    restaurants.append(restaurant_info)
                
                # Sort by rating and number of reviews
                restaurants.sort(key=lambda x: (x['rating'], x['user_ratings_total']), reverse=True)
                
                execution_time = time.time() - start_time
                api_details = {
                    "location_geocoded": lat_lng,
                    "restaurants_found": len(restaurants),
                    "api_call": "searchNearby_restaurants"
                }
                
                tool_logger.log_execution(
                    "restaurant_search", input_data, restaurants, 
                    execution_time, True, None, api_details
                )
                
                return restaurants
            else:
                # Fallback to text search
                return self._search_restaurants_by_text(location, price_level_max)
                
        except Exception as e:
            error_msg = f"Failed to search restaurants: {str(e)}"
            execution_time = time.time() - start_time
            output_data = [{"error": error_msg}]
            tool_logger.log_execution(
                "restaurant_search", input_data, output_data, 
                execution_time, False, error_msg
            )
            return output_data
    
    def _search_restaurants_by_text(self, location: str, price_level_max: int = 3) -> List[Dict]:
        """Search restaurants using text search when geocoding fails"""
        try:
            url = f"{self.places_base_url}/places:searchText"
            headers = {
                'Content-Type': 'application/json',
                'X-Goog-Api-Key': self.api_key,
                'X-Goog-FieldMask': 'places.id,places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.priceLevel,places.types,places.location,places.editorialSummary,places.currentOpeningHours'
            }
            
            data = {
                "textQuery": f"restaurants near {location}",
                "maxResultCount": 10,
                "includedType": "restaurant",
                "rankPreference": "RELEVANCE"
            }
            
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                restaurants = []
                
                for place in result.get('places', []):
                    place_price_level = self._extract_price_level(place)
                    if place_price_level > price_level_max:
                        continue
                    
                    restaurant_info = {
                        'name': place.get('displayName', {}).get('text', 'Unknown'),
                        'address': place.get('formattedAddress', ''),
                        'rating': place.get('rating', 0),
                        'user_ratings_total': place.get('userRatingCount', 0),
                        'price_level': place_price_level,
                        'price_level_text': self._get_price_level_text(place_price_level),
                        'types': place.get('types', []),
                        'place_id': place.get('id', ''),
                        'location': place.get('location', {}),
                        'editorial_summary': place.get('editorialSummary', {}).get('text', ''),
                        'opening_hours': self._extract_opening_hours(place),
                        'cuisine_type': self._extract_cuisine_type(place),
                        'recommended_for': self._get_meal_recommendation(place)
                    }
                    restaurants.append(restaurant_info)
                
                restaurants.sort(key=lambda x: (x['rating'], x['user_ratings_total']), reverse=True)
                return restaurants
            else:
                return [{"error": f"Restaurant text search failed: {response.status_code}"}]
                
        except Exception as e:
            return [{"error": f"Restaurant text search failed: {str(e)}"}]
    
    def _get_price_level_text(self, price_level: int) -> str:
        """Convert price level to human-readable text"""
        price_text = {
            0: "Free",
            1: "Budget-friendly ($)",
            2: "Moderate ($$)",
            3: "Expensive ($$$)",
            4: "Very Expensive ($$$$)"
        }
        return price_text.get(price_level, "Unknown")
    
    def _extract_cuisine_type(self, place_data: Dict) -> str:
        """Extract cuisine type from place types"""
        types = place_data.get('types', [])
        
        # Common cuisine types in Google Places
        cuisine_types = {
            'japanese_restaurant': 'Japanese',
            'chinese_restaurant': 'Chinese',
            'italian_restaurant': 'Italian',
            'french_restaurant': 'French',
            'korean_restaurant': 'Korean',
            'thai_restaurant': 'Thai',
            'indian_restaurant': 'Indian',
            'mexican_restaurant': 'Mexican',
            'american_restaurant': 'American',
            'seafood_restaurant': 'Seafood',
            'steak_house': 'Steakhouse',
            'sushi_restaurant': 'Sushi',
            'ramen_restaurant': 'Ramen',
            'barbecue_restaurant': 'BBQ',
            'pizza_restaurant': 'Pizza',
            'cafe': 'Cafe',
            'bakery': 'Bakery'
        }
        
        for type_key in types:
            if type_key in cuisine_types:
                return cuisine_types[type_key]
        
        # If no specific cuisine found, return general
        return "Various"
    
    def _get_meal_recommendation(self, place_data: Dict) -> str:
        """Recommend which meal this restaurant is best for"""
        types = place_data.get('types', [])
        
        # Breakfast/brunch places
        if any(t in types for t in ['cafe', 'bakery', 'breakfast_restaurant']):
            return "Breakfast/Brunch"
        
        # Bars and evening venues
        if any(t in types for t in ['bar', 'night_club']):
            return "Dinner/Evening"
        
        # Check opening hours for more specific recommendations
        opening_hours = place_data.get('currentOpeningHours', {})
        if opening_hours:
            # This would require more complex parsing of opening hours
            # For now, return general recommendation
            return "Lunch/Dinner"
        
        return "Any meal"
    
    def search_pois(self, location: str, radius: int = 10000, poi_type: str = "tourist_attraction") -> List[Dict]:
        """Search for POIs near a location using Places API (New)"""
        try:
            url = f"{self.places_base_url}/places:searchNearby"
            headers = {
                'Content-Type': 'application/json',
                'X-Goog-Api-Key': self.api_key,
                'X-Goog-FieldMask': 'places.id,places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.types,places.location'
            }
            
            # Try to geocode the location first
            geocode_result = self._geocode_location(location)
            if 'error' in geocode_result:
                return [{"error": f"Failed to geocode location '{location}': {geocode_result['error']}"}]
            
            lat_lng = geocode_result['location']
            
            data = {
                "includedTypes": [poi_type],
                "maxResultCount": 20,
                "locationRestriction": {
                    "circle": {
                        "center": {
                            "latitude": lat_lng['lat'],
                            "longitude": lat_lng['lng']
                        },
                        "radius": radius
                    }
                }
            }
            
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                pois = []
                
                for place in result.get('places', []):
                    poi_info = {
                        'name': place.get('displayName', {}).get('text', 'Unknown'),
                        'address': place.get('formattedAddress', ''),
                        'rating': place.get('rating', 0),
                        'user_ratings_total': place.get('userRatingCount', 0),
                        'types': place.get('types', []),
                        'place_id': place.get('id', ''),
                        'location': place.get('location', {}),
                        'importance_score': self._calculate_importance_score(place)
                    }
                    pois.append(poi_info)
                
                return pois
            else:
                return [{"error": f"Search failed: {response.status_code} - {response.text}"}]
                
        except Exception as e:
            return [{"error": f"Failed to search POIs: {str(e)}"}]
    
    def _geocode_location(self, location: str) -> Dict:
        """Geocode a location string to lat/lng using Geocoding API"""
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'address': location,
            'key': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                result = response.json()
                if result.get('results'):
                    location_data = result['results'][0]['geometry']['location']
                    return {'location': location_data}
                else:
                    return {"error": "Location not found"}
            else:
                return {"error": f"Geocoding failed: {response.status_code}"}
        except Exception as e:
            return {"error": f"Geocoding request failed: {str(e)}"}

class RouteSearchTool:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        self.routes_base_url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    
    def get_route(self, origin: str, destination: str, mode: str = "DRIVE") -> Dict:
        """Get route information using Routes API (New)"""
        start_time = time.time()
        input_data = {"origin": origin, "destination": destination, "mode": mode}
        
        try:
            headers = {
                'Content-Type': 'application/json',
                'X-Goog-Api-Key': self.api_key,
                'X-Goog-FieldMask': 'routes.duration,routes.distanceMeters,routes.legs,routes.polyline.encodedPolyline'
            }
            
            # Map old mode to new travel mode
            travel_mode_mapping = {
                'driving': 'DRIVE',
                'walking': 'WALK',
                'bicycling': 'BICYCLE',
                'transit': 'TRANSIT'
            }
            travel_mode = travel_mode_mapping.get(mode.lower(), mode.upper())
            
            # Debug logging
            print(f"🔍 Route search: {origin} -> {destination}")
            print(f"   Mode: {travel_mode}")
            
            # Try to geocode addresses for better results (especially for Japanese addresses)
            origin_geocoded = self._geocode_location(origin)
            dest_geocoded = self._geocode_location(destination)
            
            # Determine if we should use geocoded coordinates or addresses
            use_geocoding = ('error' not in origin_geocoded and 'error' not in dest_geocoded)
            
            if use_geocoding:
                print(f"   Using geocoded coordinates")
                data = {
                    "origin": {
                        "location": {
                            "latLng": {
                                "latitude": origin_geocoded['location']['lat'],
                                "longitude": origin_geocoded['location']['lng']
                            }
                        }
                    },
                    "destination": {
                        "location": {
                            "latLng": {
                                "latitude": dest_geocoded['location']['lat'],
                                "longitude": dest_geocoded['location']['lng']
                            }
                        }
                    },
                    "travelMode": travel_mode,
                    "regionCode": "JP",  # Explicitly set region for Japan
                    "languageCode": "ja",  # Set language to Japanese
                    "computeAlternativeRoutes": True,
                    "routeModifiers": {
                        "avoidTolls": False,
                        "avoidHighways": False,
                        "avoidFerries": False
                    },
                    "transitPreferences": { 
                        "allowedTravelModes": ["BUS","SUBWAY","TRAIN","LIGHT_RAIL","RAIL"],
                        "routingPreference": "FEWER_TRANSFERS"
                    }
                }
            else:
                print(f"   Using original addresses (geocoding failed)")
                data = {
                    "origin": {
                        "address": origin
                    },
                    "destination": {
                        "address": destination
                    },
                    "travelMode": travel_mode,
                    "regionCode": "JP",  # Explicitly set region for Japan
                    "languageCode": "ja",  # Set language to Japanese
                    "computeAlternativeRoutes": True,
                    "routeModifiers": {
                        "avoidTolls": False,
                        "avoidHighways": False,
                        "avoidFerries": False
                    },
                    "transitPreferences": { 
                        "allowedTravelModes": ["BUS","SUBWAY","TRAIN","LIGHT_RAIL","RAIL"],
                        "routingPreference": "FEWER_TRANSFERS"
                    }
                }
            
            response = requests.post(self.routes_base_url, json=data, headers=headers)
            
            print(f"   Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Debug: print raw response
                if not result.get('routes'):
                    print(f"   ⚠️ No routes found in response")
                    print(f"   Raw response: {json.dumps(result, indent=2)}")
                else:
                    print(f"   ✅ Found {len(result.get('routes', []))} routes")
                
                route_info = {
                    'routes': [],
                    'origin': origin,
                    'destination': destination,
                    'mode': travel_mode
                }
                
                for route in result.get('routes', [])[:3]:  # Top 3 routes
                    duration = route.get('duration', '')
                    distance_meters = route.get('distanceMeters', 0)
                    
                    # Convert duration from seconds format to readable format
                    duration_text = self._format_duration(duration)
                    summary = self._get_summary(route)
                    
                    # Convert distance to readable format
                    distance_text = f"{distance_meters / 1000:.1f} km" if distance_meters > 0 else "Unknown distance"
                    
                    legs = route.get('legs', [])
                    steps_count = sum(len(leg.get('steps', [])) for leg in legs)
                    
                    route_details = {
                        'duration': duration_text,
                        'duration_value': self._parse_duration_to_seconds(duration),
                        'distance': distance_text,
                        'distance_value': distance_meters,
                        'steps': steps_count,
                        'summary': summary,
                        'warnings': []
                    }
                    route_info['routes'].append(route_details)
                
                execution_time = time.time() - start_time
                api_details = {
                    "travel_mode": travel_mode,
                    "routes_found": len(result.get('routes', [])),
                    "api_call": "computeRoutes"
                }
                
                tool_logger.log_execution(
                    "route_search", input_data, route_info, 
                    execution_time, True, None, api_details
                )
                
                # If no routes found, try legacy Directions API as fallback
                if not route_info['routes']:
                    print(f"   Trying legacy Directions API as fallback...")
                    fallback_result = self._legacy_directions_fallback(origin, destination, travel_mode)
                    if fallback_result.get('routes'):
                        route_info = fallback_result
                    else:
                        # If still no routes, try Browser Use
                        print(f"   Trying Browser Use as final fallback...")
                        browser_fallback = self._browser_use_fallback(origin, destination, travel_mode)
                        if browser_fallback.get('routes'):
                            route_info = browser_fallback
                
                return route_info
            else:
                error_msg = f"Route search failed: {response.status_code} - {response.text}"
                print(f"   ❌ {error_msg}")
                
                # Try legacy API as fallback
                print(f"   Trying legacy Directions API as fallback...")
                fallback_result = self._legacy_directions_fallback(origin, destination, travel_mode)
                if not fallback_result.get('error'):
                    execution_time = time.time() - start_time
                    tool_logger.log_execution(
                        "route_search", input_data, fallback_result, 
                        execution_time, True, None, {"fallback": "legacy_directions_api"}
                    )
                    return fallback_result
                
                # If both APIs fail, try Browser Use as final fallback
                print(f"   Trying Browser Use as final fallback...")
                browser_fallback = self._browser_use_fallback(origin, destination, travel_mode)
                if not browser_fallback.get('error'):
                    execution_time = time.time() - start_time
                    tool_logger.log_execution(
                        "route_search", input_data, browser_fallback,
                        execution_time, True, None, {"fallback": "browser_use"}
                    )
                    return browser_fallback
                
                execution_time = time.time() - start_time
                output_data = {"error": error_msg}
                tool_logger.log_execution(
                    "route_search", input_data, output_data, 
                    execution_time, False, error_msg
                )
                return output_data
                
        except Exception as e:
            error_msg = f"Failed to get route: {str(e)}"
            execution_time = time.time() - start_time
            output_data = {"error": error_msg}
            tool_logger.log_execution(
                "route_search", input_data, output_data, 
                execution_time, False, error_msg
            )
            return output_data
    
    def _format_duration(self, duration_str: str) -> str:
        """Convert duration from seconds format to readable text"""
        if not duration_str:
            return "Unknown duration"
        
        # Remove 's' from the end if present (e.g., "1234s" -> "1234")
        if duration_str.endswith('s'):
            try:
                seconds = int(duration_str[:-1])
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                
                if hours > 0:
                    return f"{hours} hour{'s' if hours != 1 else ''} {minutes} min{'s' if minutes != 1 else ''}"
                else:
                    return f"{minutes} min{'s' if minutes != 1 else ''}"
            except ValueError:
                return duration_str
        
        return duration_str
    
    def _parse_duration_to_seconds(self, duration_str: str) -> int:
        """Parse duration string to seconds"""
        if not duration_str:
            return 0
        
        if duration_str.endswith('s'):
            try:
                return int(duration_str[:-1])
            except ValueError:
                return 0
        
        return 0
    
    def _get_summary(self, route: Dict) -> str:
        """Get summary of the route"""
        try:
            legs = route.get('legs', [])
            if not legs:
                return ''
            
            steps_overview = legs[0].get('stepsOverview', {})
            multi_modal_segments = steps_overview.get('multiModalSegments', [])
            
            if not multi_modal_segments:
                return ''
            
            summary_parts = []
            for segment in multi_modal_segments:
                travel_mode = segment.get('travelMode', 'WALK')
                navigation = segment.get('navigationInstruction', {})
                instruction = navigation.get('instructions', '')
                
                if instruction:
                    summary_parts.append(f"{travel_mode}: {instruction}")
                else:
                    summary_parts.append(travel_mode)
            
            return ' → '.join(summary_parts) if summary_parts else ''
            
        except Exception as e:
            return f'Failed to get route summary: {str(e)}'
    
    def _geocode_location(self, location: str) -> Dict:
        """Geocode a location string to lat/lng using Geocoding API"""
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'address': location,
            'key': self.api_key,
            'region': 'jp'  # Prioritize Japanese results
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                result = response.json()
                if result.get('results'):
                    location_data = result['results'][0]['geometry']['location']
                    return {'location': location_data}
                else:
                    return {"error": "Location not found"}
            else:
                return {"error": f"Geocoding failed: {response.status_code}"}
        except Exception as e:
            return {"error": f"Geocoding request failed: {str(e)}"}
    
    def _browser_use_fallback(self, origin: str, destination: str, mode: str) -> Dict:
        """Fallback to Browser Use when APIs fail"""
        try:
            browser_tool = BrowserUseRouteSearch()
            result = browser_tool.search_route_sync(origin, destination, mode)
            return result
        except Exception as e:
            error_msg = f"Browser Use fallback failed: {str(e)}"
            if "No such file or directory" in str(e) and "playwright" in str(e).lower():
                error_msg += "\n\nNote: Playwright browsers are not installed. Please run: playwright install chromium"
            return {"error": error_msg}
    
    def _legacy_directions_fallback(self, origin: str, destination: str, mode: str) -> Dict:
        """Fallback to legacy Directions API when Routes API fails"""
        try:
            url = "https://maps.googleapis.com/maps/api/directions/json"
            
            # Map new mode to legacy mode
            mode_mapping = {
                'DRIVE': 'driving',
                'WALK': 'walking',
                'BICYCLE': 'bicycling',
                'TRANSIT': 'transit'
            }
            legacy_mode = mode_mapping.get(mode, 'driving')
            
            params = {
                'origin': origin,
                'destination': destination,
                'mode': legacy_mode,
                'key': self.api_key,
                'alternatives': 'true',
                'language': 'ja',
                'region': 'jp'
            }
            
            response = requests.get(url, params=params)
            print(f"   Legacy API status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('status') == 'OK' and result.get('routes'):
                    print(f"   ✅ Legacy API found {len(result['routes'])} routes")
                    
                    route_info = {
                        'routes': [],
                        'origin': origin,
                        'destination': destination,
                        'mode': mode,
                        'api_used': 'legacy_directions'
                    }
                    
                    for route in result['routes'][:3]:
                        leg = route['legs'][0] if route.get('legs') else {}
                        
                        route_details = {
                            'duration': leg.get('duration', {}).get('text', 'Unknown'),
                            'duration_value': leg.get('duration', {}).get('value', 0),
                            'distance': leg.get('distance', {}).get('text', 'Unknown'),
                            'distance_value': leg.get('distance', {}).get('value', 0),
                            'steps': len(leg.get('steps', [])),
                            'summary': route.get('summary', ''),
                            'warnings': route.get('warnings', [])
                        }
                        route_info['routes'].append(route_details)
                    
                    return route_info
                else:
                    print(f"   ❌ Legacy API status: {result.get('status', 'UNKNOWN')}")
                    return {"error": f"Legacy API failed: {result.get('status', 'UNKNOWN')}"}
            else:
                return {"error": f"Legacy API request failed: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Legacy API exception: {str(e)}"}

class AccommodationSearchTool:
    def __init__(self):
        self.api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        self.places_base_url = "https://places.googleapis.com/v1"
    
    def search_accommodations(self, location: str, checkin_date: str = None, checkout_date: str = None) -> List[Dict]:
        """Search for accommodations using Places API (New)"""
        start_time = time.time()
        input_data = {"location": location, "checkin_date": checkin_date, "checkout_date": checkout_date}
        
        try:
            url = f"{self.places_base_url}/places:searchNearby"
            headers = {
                'Content-Type': 'application/json',
                'X-Goog-Api-Key': self.api_key,
                'X-Goog-FieldMask': 'places.id,places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.priceLevel,places.types,places.location,places.editorialSummary,places.websiteUri,places.nationalPhoneNumber,places.googleMapsUri'
            }
            
            # Try to geocode the location first
            geocode_result = self._geocode_location(location)
            if 'error' in geocode_result:
                # If geocoding fails, try text search instead
                return self._search_accommodations_by_text(location, checkin_date, checkout_date)
            
            lat_lng = geocode_result['location']
            
            data = {
                "includedTypes": ["lodging", "hotel", "motel", "resort_hotel"],
                "maxResultCount": 20,
                "locationRestriction": {
                    "circle": {
                        "center": {
                            "latitude": lat_lng['lat'],
                            "longitude": lat_lng['lng']
                        },
                        "radius": 5000
                    }
                },
                "rankPreference": "POPULARITY"
            }
            
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                accommodations = []
                
                for place in result.get('places', []):
                    # Extract price level
                    price_level = self._extract_price_level(place)
                    
                    acc_info = {
                        'name': place.get('displayName', {}).get('text', 'Unknown'),
                        'address': place.get('formattedAddress', ''),
                        'rating': place.get('rating', 0),
                        'user_ratings_total': place.get('userRatingCount', 0),
                        'price_level': price_level,
                        'price_level_text': self._get_price_level_text(price_level),
                        'types': place.get('types', []),
                        'place_id': place.get('id', ''),
                        'location': place.get('location', {}),
                        'editorial_summary': place.get('editorialSummary', {}).get('text', ''),
                        'website': place.get('websiteUri', ''),
                        'phone': place.get('nationalPhoneNumber', ''),
                        'google_maps_url': place.get('googleMapsUri', ''),
                        'accommodation_type': self._extract_accommodation_type(place)
                    }
                    accommodations.append(acc_info)
                
                # Sort by rating and number of reviews
                accommodations.sort(key=lambda x: (x['rating'], x['user_ratings_total']), reverse=True)
                
                execution_time = time.time() - start_time
                api_details = {
                    "location_geocoded": lat_lng,
                    "accommodations_found": len(accommodations),
                    "api_call": "searchNearby"
                }
                
                tool_logger.log_execution(
                    "accommodation_search", input_data, accommodations, 
                    execution_time, True, None, api_details
                )
                
                return accommodations
            else:
                # Fallback to text search
                return self._search_accommodations_by_text(location, checkin_date, checkout_date)
                
        except Exception as e:
            error_msg = f"Failed to search accommodations: {str(e)}"
            execution_time = time.time() - start_time
            output_data = [{"error": error_msg}]
            tool_logger.log_execution(
                "accommodation_search", input_data, output_data, 
                execution_time, False, error_msg
            )
            return output_data
    
    def _geocode_location(self, location: str) -> Dict:
        """Geocode a location string to lat/lng using Geocoding API"""
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            'address': location,
            'key': self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                result = response.json()
                if result.get('results'):
                    location_data = result['results'][0]['geometry']['location']
                    return {'location': location_data}
                else:
                    return {"error": "Location not found"}
            else:
                return {"error": f"Geocoding failed: {response.status_code}"}
        except Exception as e:
            return {"error": f"Geocoding request failed: {str(e)}"}
    
    def _extract_price_level(self, place_data: Dict) -> int:
        """Extract price level from place data"""
        price_level = place_data.get('priceLevel', 'PRICE_LEVEL_UNSPECIFIED')
        price_mapping = {
            'PRICE_LEVEL_FREE': 0,
            'PRICE_LEVEL_INEXPENSIVE': 1,
            'PRICE_LEVEL_MODERATE': 2,
            'PRICE_LEVEL_EXPENSIVE': 3,
            'PRICE_LEVEL_VERY_EXPENSIVE': 4,
            'PRICE_LEVEL_UNSPECIFIED': 0
        }
        return price_mapping.get(price_level, 0)
    
    def _get_price_level_text(self, price_level: int) -> str:
        """Convert price level to human-readable text"""
        price_text = {
            0: "Free",
            1: "Budget-friendly ($)",
            2: "Moderate ($$)",
            3: "Expensive ($$$)",
            4: "Very Expensive ($$$$)"
        }
        return price_text.get(price_level, "Unknown")
    
    def _extract_accommodation_type(self, place_data: Dict) -> str:
        """Extract accommodation type from place types"""
        types = place_data.get('types', [])
        
        # Common accommodation types in Google Places
        accommodation_types = {
            'hotel': 'Hotel',
            'motel': 'Motel',
            'resort_hotel': 'Resort Hotel',
            'extended_stay_hotel': 'Extended Stay Hotel',
            'budget_hotel': 'Budget Hotel',
            'bed_and_breakfast': 'Bed & Breakfast',
            'guest_house': 'Guest House',
            'hostel': 'Hostel',
            'lodging': 'Lodging',
            'spa': 'Spa Resort',
            'inn': 'Inn',
            'cottage': 'Cottage',
            'apartment': 'Serviced Apartment',
            'resort': 'Resort'
        }
        
        for type_key in types:
            if type_key in accommodation_types:
                return accommodation_types[type_key]
        
        # If no specific type found, return general
        return "Hotel"
    
    def _search_accommodations_by_text(self, location: str, checkin_date: str = None, checkout_date: str = None) -> List[Dict]:
        """Search accommodations using text search when geocoding fails"""
        start_time = time.time()
        input_data = {"location": location, "checkin_date": checkin_date, "checkout_date": checkout_date}
        
        try:
            url = f"{self.places_base_url}/places:searchText"
            headers = {
                'Content-Type': 'application/json',
                'X-Goog-Api-Key': self.api_key,
                'X-Goog-FieldMask': 'places.id,places.displayName,places.formattedAddress,places.rating,places.userRatingCount,places.priceLevel,places.types,places.location,places.editorialSummary,places.websiteUri,places.nationalPhoneNumber,places.googleMapsUri'
            }
            
            data = {
                "textQuery": f"hotels in {location}",
                "maxResultCount": 20,
                "includedType": "lodging",
                "rankPreference": "RELEVANCE"
            }
            
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                accommodations = []
                
                for place in result.get('places', []):
                    price_level = self._extract_price_level(place)
                    
                    acc_info = {
                        'name': place.get('displayName', {}).get('text', 'Unknown'),
                        'address': place.get('formattedAddress', ''),
                        'rating': place.get('rating', 0),
                        'user_ratings_total': place.get('userRatingCount', 0),
                        'price_level': price_level,
                        'price_level_text': self._get_price_level_text(price_level),
                        'types': place.get('types', []),
                        'place_id': place.get('id', ''),
                        'location': place.get('location', {}),
                        'editorial_summary': place.get('editorialSummary', {}).get('text', ''),
                        'website': place.get('websiteUri', ''),
                        'phone': place.get('nationalPhoneNumber', ''),
                        'google_maps_url': place.get('googleMapsUri', ''),
                        'accommodation_type': self._extract_accommodation_type(place)
                    }
                    accommodations.append(acc_info)
                
                # Sort by rating and number of reviews
                accommodations.sort(key=lambda x: (x['rating'], x['user_ratings_total']), reverse=True)
                
                execution_time = time.time() - start_time
                api_details = {
                    "search_query": f"hotels in {location}",
                    "accommodations_found": len(accommodations),
                    "api_call": "searchText"
                }
                
                tool_logger.log_execution(
                    "accommodation_text_search", input_data, accommodations, 
                    execution_time, True, None, api_details
                )
                
                return accommodations
            else:
                error_msg = f"Accommodation text search failed: {response.status_code}"
                execution_time = time.time() - start_time
                output_data = [{"error": error_msg}]
                tool_logger.log_execution(
                    "accommodation_text_search", input_data, output_data, 
                    execution_time, False, error_msg
                )
                return output_data
                
        except Exception as e:
            error_msg = f"Accommodation text search failed: {str(e)}"
            execution_time = time.time() - start_time
            output_data = [{"error": error_msg}]
            tool_logger.log_execution(
                "accommodation_text_search", input_data, output_data, 
                execution_time, False, error_msg
            )
            return output_data

class BrowserUseRouteSearch:
    """Browser Useを使用してGoogle Mapsで経路探索を行うツール"""
    
    def __init__(self):
        pass
    
    async def search_route_with_browser(self, origin: str, destination: str, mode: str = "DRIVE") -> Dict:
        """Browser Useを使用してGoogle Mapsで経路探索を実行"""
        start_time = time.time()
        input_data = {"origin": origin, "destination": destination, "mode": mode}
        
        try:
            from browser_use import Agent
            from browser_use.llm import ChatOpenAI
            from browser_use.browser.browser import Browser, BrowserConfig
            import asyncio
            
            # モードをGoogle Maps用に変換
            mode_mapping = {
                'DRIVE': 'driving',
                'WALK': 'walking', 
                'BICYCLE': 'cycling',
                'TRANSIT': 'transit'
            }
            google_mode = mode_mapping.get(mode.upper(), 'driving')
            
            print(f"🌐 Browser Use: Searching route {origin} -> {destination} (mode: {google_mode})")
            
            # タスクの詳細な指示
            detailed_task = f"""
            Go to Google Maps and find the route from '{origin}' to '{destination}' using {google_mode} mode.
            Note: Set the departure time to 11:00 AM.
            Please extract:
            1. The duration of the route
            2. The distance of the route  
            3. Main route description or steps
            4. If multiple routes exist, get information for up to 3 routes
            
            Return the information in a structured format with duration, distance, and route summary.
            """
            
            browser = Browser(
                config=BrowserConfig(
                    headless=True,
                    browser_type="chromium",
                    browser_version="120.0.0"
                )
            )
            
            # LLMを作成
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            
            # エージェントを作成して実行
            agent = Agent(
                task=detailed_task,
                llm=llm,
                browser=browser,
            )
            
            # エージェントを実行
            result = await agent.run()
            
            # 結果を解析してAPI形式に変換
            route_info = self._parse_browser_result(result, origin, destination, mode)
            
            execution_time = time.time() - start_time
            api_details = {
                "method": "browser_use",
                "google_mode": google_mode,
                "task_result": str(result)[:500]  # 最初の500文字のみ
            }
            
            tool_logger.log_execution(
                "browser_use_route_search", input_data, route_info,
                execution_time, True, None, api_details
            )
            
            return route_info
            
        except Exception as e:
            error_msg = f"Browser Use route search failed: {str(e)}"
            
            # Check if it's a Playwright installation issue
            if "No such file or directory" in str(e) and "playwright" in str(e).lower():
                error_msg += "\n\nNote: Playwright browsers are not installed. Please run: playwright install chromium"
            
            execution_time = time.time() - start_time
            output_data = {"error": error_msg}
            tool_logger.log_execution(
                "browser_use_route_search", input_data, output_data,
                execution_time, False, error_msg
            )
            return output_data
    
    def _parse_browser_result(self, result: Any, origin: str, destination: str, mode: str) -> Dict:
        """Browser Useの結果をAPI形式に変換"""
        try:
            # デフォルトの結果構造
            route_info = {
                'routes': [],
                'origin': origin,
                'destination': destination,
                'mode': mode,
                'api_used': 'browser_use'
            }
            
            # resultから情報を抽出（Browser Useの出力形式に依存）
            # この部分は実際のBrowser Useの出力形式に合わせて調整が必要
            if isinstance(result, dict):
                # 単一ルートの場合
                if 'duration' in result or 'distance' in result:
                    route_details = {
                        'duration': result.get('duration', 'Unknown'),
                        'duration_value': self._parse_duration_text(result.get('duration', '')),
                        'distance': result.get('distance', 'Unknown'),
                        'distance_value': self._parse_distance_text(result.get('distance', '')),
                        'steps': result.get('steps', 0),
                        'summary': result.get('summary', 'Route via Browser Use'),
                        'warnings': []
                    }
                    route_info['routes'].append(route_details)
                
                # 複数ルートの場合
                elif 'routes' in result:
                    for route in result['routes'][:3]:
                        route_details = {
                            'duration': route.get('duration', 'Unknown'),
                            'duration_value': self._parse_duration_text(route.get('duration', '')),
                            'distance': route.get('distance', 'Unknown'),
                            'distance_value': self._parse_distance_text(route.get('distance', '')),
                            'steps': route.get('steps', 0),
                            'summary': route.get('summary', 'Route via Browser Use'),
                            'warnings': []
                        }
                        route_info['routes'].append(route_details)
            
            # resultが文字列の場合、パースを試みる
            elif isinstance(result, str):
                # 簡易的なパース（実際の出力に応じて改善が必要）
                route_details = {
                    'duration': 'Unknown (Browser result)',
                    'duration_value': 0,
                    'distance': 'Unknown (Browser result)',
                    'distance_value': 0,
                    'steps': 0,
                    'summary': f'Browser Use result: {result[:200]}...',
                    'warnings': ['Route information extracted from browser']
                }
                route_info['routes'].append(route_details)
            
            return route_info
            
        except Exception as e:
            # エラー時でも何か返す
            return {
                'routes': [{
                    'duration': 'Unknown',
                    'duration_value': 0,
                    'distance': 'Unknown',
                    'distance_value': 0,
                    'steps': 0,
                    'summary': f'Browser parsing error: {str(e)}',
                    'warnings': ['Failed to parse browser result']
                }],
                'origin': origin,
                'destination': destination,
                'mode': mode,
                'api_used': 'browser_use',
                'error': f'Result parsing failed: {str(e)}'
            }
    
    def _parse_duration_text(self, duration_text: str) -> int:
        """時間テキストを秒数に変換"""
        try:
            # "1 hour 30 mins" -> 5400秒
            # "45 mins" -> 2700秒
            total_seconds = 0
            
            if 'hour' in duration_text:
                hours = int(duration_text.split('hour')[0].strip().split()[-1])
                total_seconds += hours * 3600
            
            if 'min' in duration_text:
                mins_part = duration_text.split('min')[0]
                if 'hour' in mins_part:
                    mins_part = mins_part.split('hour')[1]
                mins = int(mins_part.strip().split()[-1])
                total_seconds += mins * 60
            
            return total_seconds
            
        except:
            return 0
    
    def _parse_distance_text(self, distance_text: str) -> int:
        """距離テキストをメートルに変換"""
        try:
            # "15.5 km" -> 15500
            # "500 m" -> 500
            if 'km' in distance_text:
                km = float(distance_text.replace('km', '').strip())
                return int(km * 1000)
            elif 'm' in distance_text:
                return int(distance_text.replace('m', '').strip())
            
            return 0
            
        except:
            return 0
    
    def search_route_sync(self, origin: str, destination: str, mode: str = "DRIVE") -> Dict:
        """同期的にBrowser Useを実行（LangChainツール用）"""
        try:
            import asyncio
            
            # 既存のイベントループがある場合の処理
            try:
                loop = asyncio.get_running_loop()
                # 既にイベントループ内にいる場合は、新しいタスクとして実行
                import nest_asyncio
                nest_asyncio.apply()
                return asyncio.run(self.search_route_with_browser(origin, destination, mode))
            except RuntimeError:
                # イベントループがない場合は、新しく作成
                return asyncio.run(self.search_route_with_browser(origin, destination, mode))
                
        except Exception as e:
            return {
                "error": f"Browser Use sync execution failed: {str(e)}",
                "origin": origin,
                "destination": destination,
                "mode": mode
            }

class WebSearchTool:
    def __init__(self):
        self.api_key = os.getenv('SERPER_API_KEY')  # Using Serper API for web search
    
    def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """Perform web search using Serper API"""
        try:
            url = "https://google.serper.dev/search"
            payload = {
                'q': query,
                'num': num_results
            }
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                results = response.json()
                search_results = []
                
                for result in results.get('organic', []):
                    search_results.append({
                        'title': result.get('title'),
                        'link': result.get('link'),
                        'snippet': result.get('snippet'),
                        'date': result.get('date')
                    })
                
                return search_results
            else:
                return [{"error": f"Search failed with status code: {response.status_code}"}]
        except Exception as e:
            return [{"error": f"Failed to perform web search: {str(e)}"}]

# Create tool instances
google_maps_tool = GoogleMapsAPITool()
route_search_tool = RouteSearchTool()
accommodation_search_tool = AccommodationSearchTool()
web_search_tool = WebSearchTool()
browser_use_route_tool = BrowserUseRouteSearch()

# Helper function for safe route search
def _safe_route_search(input_str: str) -> Dict:
    """Safely handle route search input with proper argument parsing"""
    try:
        parts = input_str.split('|')
        if len(parts) >= 2:
            origin = parts[0].strip()
            destination = parts[1].strip()
            mode = parts[2].strip() if len(parts) > 2 else "DRIVE"
            return route_search_tool.get_route(origin, destination, mode)
        else:
            return {"error": "Route search requires at least origin and destination separated by |"}
    except Exception as e:
        return {"error": f"Route search failed: {str(e)}"}

# Helper function for safe restaurant search
def _safe_restaurant_search(input_str: str) -> List[Dict]:
    """Safely handle restaurant search input with proper argument parsing"""
    try:
        parts = input_str.split('|')
        if len(parts) >= 1:
            location = parts[0].strip()
            radius = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1000
            price_level_max = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 3
            return google_maps_tool.search_restaurants(location, radius, price_level_max)
        else:
            return [{"error": "Restaurant search requires at least a location"}]
    except Exception as e:
        return [{"error": f"Restaurant search failed: {str(e)}"}]

# Define tools for langchain
def create_tools():
    return [
        Tool(
            name="poi_details_search",
            description="Get detailed information for a specific POI. Input should be 'poi_name|location_context' format.",
            func=lambda input_str: google_maps_tool.get_poi_details(
                *input_str.split('|', 1) if '|' in input_str else (input_str, "")
            )
        ),
        Tool(
            name="google_maps_search",
            description="Search for Points of Interest (POIs) using Google Maps API. Input should be location string.",
            func=lambda location: google_maps_tool.search_pois(location)
        ),
        Tool(
            name="route_search",
            description="Get route information between two locations. Input should be 'origin|destination|mode' format.",
            func=lambda input_str: _safe_route_search(input_str)
        ),
        Tool(
            name="accommodation_search",
            description="Search for accommodations near a location. Input should be location string.",
            func=lambda location: accommodation_search_tool.search_accommodations(location)
        ),
        Tool(
            name="restaurant_search",
            description="Search for restaurants near a location. Input format: 'location|radius_meters|max_price_level'. Example: 'Tokyo Tower|1000|3'. Price levels: 1=$, 2=$$, 3=$$$, 4=$$$$",
            func=lambda input_str: _safe_restaurant_search(input_str)
        ),
        Tool(
            name="web_search",
            description="Perform general web search. Input should be search query string.",
            func=lambda query: web_search_tool.search(query)
        ),
        Tool(
            name="browser_use_route_search",
            description="Search routes using Browser Use (GUI-based Google Maps). Use when API fails. Input: 'origin|destination|mode'.",
            func=lambda input_str: browser_use_route_tool.search_route_sync(
                *input_str.split('|', 2) if '|' in input_str else (input_str, "", "DRIVE")
            )
        )
    ] 