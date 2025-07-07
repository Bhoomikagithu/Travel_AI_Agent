import streamlit as st
from textwrap import dedent
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_community.utilities import SerpAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_types import AgentType
from datetime import datetime, timedelta
import uuid
import os
from dotenv import load_dotenv

# Optional imports with error handling
try:
    import pdfkit
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from ics import Calendar, Event
    ICS_AVAILABLE = True
except ImportError:
    ICS_AVAILABLE = False

try:
    import folium
    from streamlit_folium import st_folium
    MAP_AVAILABLE = True
except ImportError:
    MAP_AVAILABLE = False

load_dotenv()

st.set_page_config(page_title="✈️ AI Travel Planner", layout="centered")
st.title("AI Travel Planner ✈️ with Researcher & Planner Agents")
st.caption("Plan your trip with GPT-4.1 via GitHub Models, LangChain, SerpAPI !")

# Initialize session state
if "trip_history" not in st.session_state:
    st.session_state.trip_history = []
if "research_results" not in st.session_state:
    st.session_state.research_results = None
if "current_trip_data" not in st.session_state:
    st.session_state.current_trip_data = None
if "show_preferences" not in st.session_state:
    st.session_state.show_preferences = False

# --- Inputs ---
github_token = os.getenv("GITHUB_TOKEN")
serp_api_key = os.getenv("SERPAPI_API_KEY")

if not github_token or not serp_api_key:
    st.warning("Please set your GITHUB_TOKEN and SERPAPI_API_KEY in the .env file.")

destination = st.text_input("Where do you want to go?")
num_days = st.number_input("How many days do you want to travel for?", min_value=1, max_value=30, value=7)
budget = st.number_input("Budget (total amount in INR)", min_value=5000, max_value=5000000, value=100000, step=5000)
language = st.selectbox("Preferred language", ["English", "Hindi", "Spanish", "French", "German"])

generate_clicked = st.button("🔍 Start Research & Planning")

if generate_clicked:
    if not (github_token and serp_api_key):
        st.error("Please set your GITHUB_TOKEN and SERPAPI_API_KEY in the .env file.")
    elif not destination:
        st.error("Please enter your travel destination.")
    else:
        # Use GitHub Models instead of OpenAI API
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=github_token,
            openai_api_base="https://models.inference.ai.azure.com"
        )
        
        search = SerpAPIWrapper(serpapi_api_key=serp_api_key)
        search_tool = Tool(
            name="search_google",
            func=search.run,
            description="Searches for travel info related to destinations and activities."
        )
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        researcher_prompt = dedent(
            f"""
            CRITICAL: Respond ONLY in {language} language. All research findings must be presented in {language}.
            
            You are a world-class travel researcher. Given a travel destination and the number of days the user wants to travel for,
            generate a list of 3 search terms for relevant travel activities and accommodations.
            Then search the web for each term, analyze the results, and return the most relevant results.
            The user has a ₹{budget} INR budget and prefers the itinerary in {language}.
            
            Generate 3 search terms related to the destination and days.
            For each, use `search_google` to fetch results and analyze.
            Return 10 most relevant insights aligned with user's preferences.
            Maintain high quality and relevance.
            
            Present your findings in this format:
            
            **ACCOMMODATIONS OPTIONS:**
            - Option 1: [Name] - [Brief description] - [Price range in INR within ₹{budget} total budget]
            - Option 2: [Name] - [Brief description] - [Price range in INR within ₹{budget} total budget]
            - Option 3: [Name] - [Brief description] - [Price range in INR within ₹{budget} total budget]
            
            **ACTIVITY OPTIONS:**
            - Option 1: [Activity] - [Description] - [Duration/Time needed] - [Cost in INR]
            - Option 2: [Activity] - [Description] - [Duration/Time needed] - [Cost in INR]
            - Option 3: [Activity] - [Description] - [Duration/Time needed] - [Cost in INR]
            
            **DINING OPTIONS:**
            - Option 1: [Restaurant/Food] - [Cuisine type] - [Price range in INR]
            - Option 2: [Restaurant/Food] - [Cuisine type] - [Price range in INR]
            - Option 3: [Restaurant/Food] - [Cuisine type] - [Price range in INR]
            
            The user has a total budget of ₹{budget} INR for {num_days} days and prefers information in {language}.
            REMEMBER: Write ALL content in {language}, not English or any other language.
            """
        )
        researcher_instructions = [
            "Generate 3 search terms related to the destination and days.",
            "For each, use `search_google` to fetch results and analyze.",
            "Return 10 most relevant insights aligned with user's preferences.",
            "Maintain high quality and relevance.",
        ]

        planner_prompt = dedent(
            f"""
            You are a senior travel planner. Given destination, number of days, budget, preferred language,
            and a list of research results, create a draft itinerary.
            Include activities, accommodation suggestions, and food recommendations.
            Ensure clarity, coherence, and quality.
            """
        )
        planner_instructions = [
            "Generate a detailed day-wise itinerary.",
            "Quote facts from research results, do not fabricate.",
            "Make it engaging and tailored to user's budget and language.",
        ]

        # Setup researcher agent with search tool
        researcher = initialize_agent(
            tools=[search_tool],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            memory=memory,
            max_iterations=3,
            handle_parsing_errors=True,
        )

        with st.spinner("🔍 Researching your destination..."):
            research_input = (
                researcher_prompt
                + f"\nDestination: {destination}\nDays: {num_days}\nBudget: {budget}\nLanguage: {language}"
            )
            research_results = researcher.invoke({"input": research_input})["output"]
            
            # Store research results and trip data in session state
            st.session_state.research_results = research_results
            st.session_state.current_trip_data = {
                "destination": destination,
                "num_days": num_days,
                "budget": budget,
                "language": language
            }
            st.session_state.show_preferences = True

        st.success("✓ Research completed! Now select your preferences below.")
        st.rerun()

# Show research results and preference selection if research is completed
if st.session_state.show_preferences and st.session_state.research_results is not None:
    trip_data = st.session_state.current_trip_data
    
    st.success("✓ Research completed - Now customize your preferences!")
    
    # Display research results in organized sections
    st.subheader("🔍 Research Results")
    st.write(st.session_state.research_results)
    
    # User preference selection with detailed options and budget-based recommendations
    st.subheader("🎯 Select Your Preferences")
    st.markdown("Choose your preferred options based on the research results above:")
    
    # Accommodation Selection with box-based input
    st.markdown("### 🏨 **Accommodation Preference**")
    accommodation_options = [
        "Luxury Hotels (₹15,000-40,000+ per night)",
        "Mid-range Hotels (₹4,000-15,000 per night)", 
        "Budget Hotels/Hostels (₹800-4,000 per night)",
        "Local Guesthouses (₹1,500-6,000 per night)",
        "Vacation Rentals (₹2,500-20,000 per night)",
        "No Preference"
    ]
    
    accommodation_descriptions = {
        "Luxury Hotels (₹15,000-40,000+ per night)": "Premium hotels with high-end amenities, concierge services, and prime locations",
        "Mid-range Hotels (₹4,000-15,000 per night)": "Comfortable hotels with good amenities, central locations, and reliable service",
        "Budget Hotels/Hostels (₹800-4,000 per night)": "Basic accommodations, shared facilities, perfect for budget travelers",
        "Local Guesthouses (₹1,500-6,000 per night)": "Authentic local experience, family-run establishments, cultural immersion",
        "Vacation Rentals (₹2,500-20,000 per night)": "Apartments, houses, or condos with kitchen facilities and more space",
        "No Preference": "Let the AI choose the best option based on your budget and destination"
    }
    
    accommodation_pref = st.selectbox(
        "Select your accommodation style:",
        accommodation_options,
        index=1,
        help="Choose based on your budget and comfort preferences"
    )
    st.info(f"ℹ️ {accommodation_descriptions[accommodation_pref]}")
    
    # Show specific accommodation recommendations based on selection
    if accommodation_pref != "No Preference":
        st.markdown("#### 🏨 **Top Accommodation Recommendations for Your Selection:**")
        
        with st.spinner("🔍 Finding specific accommodation options..."):
            # Use LLM to get specific recommendations based on research and preference
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key=github_token,
                openai_api_base="https://models.inference.ai.azure.com"
            )
            
            accommodation_search_prompt = f"""
            IMPORTANT: Respond ONLY in {trip_data['language']} language. All responses must be in {trip_data['language']}.
            
            Based on the research results and user preference for {accommodation_pref} in {trip_data['destination']}, 
            provide exactly 3 specific accommodation recommendations in this format:
            
            **Option 1: [Hotel Name]**
            - Description: [Brief description and key features]
            - Price: ₹[amount] per night
            - Location: [area/location]
            - Why recommended: [reason it fits preference]
            
            **Option 2: [Hotel Name]**
            - Description: [Brief description and key features]  
            - Price: ₹[amount] per night
            - Location: [area/location]
            - Why recommended: [reason it fits preference]
            
            **Option 3: [Hotel Name]**
            - Description: [Brief description and key features]
            - Price: ₹[amount] per night  
            - Location: [area/location]
            - Why recommended: [reason it fits preference]
            
            Research context: {st.session_state.research_results}
            Budget: ₹{trip_data['budget']} INR total for {trip_data['num_days']} days
            """
            
            accommodation_recommendations = llm.invoke(accommodation_search_prompt).content
            st.write(accommodation_recommendations)
            
        # Allow user to select their preferred accommodation option
        st.markdown("##### Choose your preferred accommodation:")
        selected_accommodation = st.selectbox(
            "Select your accommodation:",
            ["Option 1", "Option 2", "Option 3", "Let AI decide based on budget"],
            key="accommodation_selection",
            help="Choose the accommodation option that best fits your preferences"
        )
    else:
        selected_accommodation = "No Preference"
    
    # Activity Selection with box-based input
    st.markdown("### 🎯 **Activity Preference**")
    activity_options = [
        "Adventure & Sports (₹2,000-10,000 per activity)",
        "Cultural & Historical (₹500-3,000 per site)",
        "Relaxation & Wellness (₹1,500-8,000 per session)",
        "Shopping & Entertainment (₹1,000-5,000 per experience)",
        "Nature & Wildlife (₹1,000-6,000 per experience)",
        "Mix of Everything"
    ]
    
    activity_descriptions = {
        "Adventure & Sports (₹2,000-10,000 per activity)": "Hiking, water sports, extreme activities, outdoor adventures",
        "Cultural & Historical (₹500-3,000 per site)": "Museums, historical sites, cultural tours, art galleries",
        "Relaxation & Wellness (₹1,500-8,000 per session)": "Spas, beaches, yoga retreats, peaceful activities",
        "Shopping & Entertainment (₹1,000-5,000 per experience)": "Markets, malls, nightlife, shows, entertainment venues",
        "Nature & Wildlife (₹1,000-6,000 per experience)": "National parks, wildlife viewing, botanical gardens, nature trails",
        "Mix of Everything": "Balanced combination of different activity types"
    }
    
    activity_pref = st.selectbox(
        "Choose your activity focus:",
        activity_options,
        index=5,
        help="Select the type of activities you're most interested in"
    )
    st.info(f"ℹ️ {activity_descriptions[activity_pref]}")
    
    # Show specific activity recommendations
    if activity_pref != "Mix of Everything":
        st.markdown("#### 🎯 **Top Activity Recommendations for Your Selection:**")
        
        with st.spinner("🔍 Finding specific activity options..."):
            activity_search_prompt = f"""
            IMPORTANT: Respond ONLY in {trip_data['language']} language. All responses must be in {trip_data['language']}.
            
            Based on the research results and user preference for {activity_pref} in {trip_data['destination']}, 
            provide exactly 3 specific activity recommendations in this format:
            
            **Option 1: [Activity Name]**
            - Description: [What to expect and details]
            - Cost: ₹[amount] per person
            - Duration: [time needed]
            - Why recommended: [reason it fits preference]
            
            **Option 2: [Activity Name]**  
            - Description: [What to expect and details]
            - Cost: ₹[amount] per person
            - Duration: [time needed]
            - Why recommended: [reason it fits preference]
            
            **Option 3: [Activity Name]**
            - Description: [What to expect and details]
            - Cost: ₹[amount] per person
            - Duration: [time needed]
            - Why recommended: [reason it fits preference]
            
            Research context: {st.session_state.research_results}
            Budget: ₹{trip_data['budget']} INR total for {trip_data['num_days']} days
            """
            
            activity_recommendations = llm.invoke(activity_search_prompt).content
            st.write(activity_recommendations)
            
        # Allow user to select their preferred activity options
        st.markdown("##### Choose your preferred activities:")
        selected_activities = st.multiselect(
            "Select activities you want to include (you can choose multiple):",
            ["Option 1", "Option 2", "Option 3"],
            key="activity_selection",
            help="Choose one or more activities for your itinerary"
        )
        if not selected_activities:  # If no activities selected, provide default
            selected_activities = ["Option 1"]  # Default selection
    else:
        selected_activities = ["Mix of Everything"]
    
    # Dining Selection with box-based input
    st.markdown("### 🍽️ **Dining Preference**")
    dining_options = [
        "Fine Dining (₹3,500-10,000+ per meal)",
        "Local Street Food (₹200-800 per meal)",
        "Mix of Both (₹800-4,000 per meal)",
        "Vegetarian/Vegan (₹400-2,500 per meal)",
        "Budget-Friendly (₹300-1,500 per meal)",
        "No Preference"
    ]
    
    dining_descriptions = {
        "Fine Dining (₹3,500-10,000+ per meal)": "High-end restaurants, chef's specials, gourmet experiences",
        "Local Street Food (₹200-800 per meal)": "Authentic local cuisine, food markets, street vendors",
        "Mix of Both (₹800-4,000 per meal)": "Combination of upscale and local dining experiences",
        "Vegetarian/Vegan (₹400-2,500 per meal)": "Plant-based restaurants and vegetarian-friendly options",
        "Budget-Friendly (₹300-1,500 per meal)": "Affordable restaurants, local eateries, good value meals",
        "No Preference": "Let the AI recommend based on local specialties and budget"
    }
    
    dining_pref = st.selectbox(
        "Choose your dining style:",
        dining_options,
        index=2,
        help="Select your preferred dining experience and budget range"
    )
    st.info(f"ℹ️ {dining_descriptions[dining_pref]}")
    
    # Show specific dining recommendations
    if dining_pref != "No Preference":
        st.markdown("#### 🍽️ **Top Dining Recommendations for Your Selection:**")
        
        with st.spinner("🔍 Finding specific dining options..."):
            dining_search_prompt = f"""
            IMPORTANT: Respond ONLY in {trip_data['language']} language. All responses must be in {trip_data['language']}.
            
            Based on the research results and user preference for {dining_pref} in {trip_data['destination']}, 
            provide exactly 3 specific restaurant/dining recommendations in this format:
            
            **Option 1: [Restaurant Name]**
            - Cuisine: [Type of cuisine]
            - Specialties: [Signature dishes]
            - Cost: ₹[amount] per person per meal
            - Location: [area/address]
            - Why recommended: [reason it fits preference]
            
            **Option 2: [Restaurant Name]**
            - Cuisine: [Type of cuisine]  
            - Specialties: [Signature dishes]
            - Cost: ₹[amount] per person per meal
            - Location: [area/address]
            - Why recommended: [reason it fits preference]
            
            **Option 3: [Restaurant Name]**
            - Cuisine: [Type of cuisine]
            - Specialties: [Signature dishes] 
            - Cost: ₹[amount] per person per meal
            - Location: [area/address]
            - Why recommended: [reason it fits preference]
            
            Research context: {st.session_state.research_results}
            Budget: ₹{trip_data['budget']} INR total for {trip_data['num_days']} days
            """
            
            dining_recommendations = llm.invoke(dining_search_prompt).content
            st.write(dining_recommendations)
            
        # Allow user to select their preferred dining options
        st.markdown("##### Choose your preferred dining options:")
        selected_dining = st.multiselect(
            "Select restaurants you want to include (you can choose multiple):",
            ["Option 1", "Option 2", "Option 3"],
            key="dining_selection",
            help="Choose one or more dining options for your itinerary"
        )
        if not selected_dining:  # If no dining selected, provide default
            selected_dining = ["Option 1"]  # Default selection
    else:
        selected_dining = ["No Preference"]
    
    # Transportation Selection with box-based input
    st.markdown("### 🚗 **Transportation Preference**")
    transport_options = [
        "Public Transport (₹50-500 per day)",
        "Rental Car (₹2,000-6,000 per day)",
        "Walking/Cycling (₹200-800 per day for rentals)",
        "Ride-sharing/Taxi (₹500-2,000 per trip)",
        "Tour Groups (₹1,500-5,000 per day)",
        "No Preference"
    ]
    
    transport_descriptions = {
        "Public Transport (₹50-500 per day)": "Buses, trains, metro - economical and authentic local experience",
        "Rental Car (₹2,000-6,000 per day)": "Freedom to explore at your own pace, access to remote locations",
        "Walking/Cycling (₹200-800 per day for rentals)": "Eco-friendly, great for short distances and city exploration",
        "Ride-sharing/Taxi (₹500-2,000 per trip)": "Convenient door-to-door service, good for specific destinations",
        "Tour Groups (₹1,500-5,000 per day)": "Organized transportation with guided experiences",
        "No Preference": "Mix of transportation methods based on convenience and cost"
    }
    
    transport_pref = st.selectbox(
        "Choose your transportation style:",
        transport_options,
        index=0,
        help="Select how you prefer to get around during your trip"
    )
    st.info(f"ℹ️ {transport_descriptions[transport_pref]}")
    
    # Show specific transportation recommendations
    if transport_pref != "No Preference":
        st.markdown("#### 🚗 **Transportation Recommendations for Your Selection:**")
        
        with st.spinner("🔍 Finding specific transportation options..."):
            transport_search_prompt = f"""
            IMPORTANT: Respond ONLY in {trip_data['language']} language. All responses must be in {trip_data['language']}.
            
            Based on the research results and user preference for {transport_pref} in {trip_data['destination']}, 
            provide exactly 3 specific transportation recommendations in this format:
            
            **Option 1: [Service Name/Type]**
            - Description: [Service details and availability]
            - Cost: ₹[amount] per day/trip
            - Coverage: [Areas served and convenience]
            - Booking: [How to book and tips]
            - Why recommended: [reason it fits preference]
            
            **Option 2: [Service Name/Type]**
            - Description: [Service details and availability]
            - Cost: ₹[amount] per day/trip  
            - Coverage: [Areas served and convenience]
            - Booking: [How to book and tips]
            - Why recommended: [reason it fits preference]
            
            **Option 3: [Service Name/Type]**
            - Description: [Service details and availability]
            - Cost: ₹[amount] per day/trip
            - Coverage: [Areas served and convenience] 
            - Booking: [How to book and tips]
            - Why recommended: [reason it fits preference]
            
            Research context: {st.session_state.research_results}
            Budget: ₹{trip_data['budget']} INR total for {trip_data['num_days']} days
            """
            
            transport_recommendations = llm.invoke(transport_search_prompt).content
            st.write(transport_recommendations)
            
        # Allow user to select their preferred transportation
        st.markdown("##### Choose your preferred transportation:")
        selected_transport = st.selectbox(
            "Select your transportation method:",
            ["Option 1", "Option 2", "Option 3", "Combination of options"],
            key="transport_selection",
            help="Choose the transportation method that works best for your trip"
        )
    else:
        selected_transport = "No Preference"
    
    # Special Requests
    st.markdown("### ✨ **Special Requests & Additional Preferences**")
    special_requests = st.text_area(
        "Any special requirements or interests?",
        placeholder="e.g., family-friendly activities, romantic spots, photography locations, accessibility needs, dietary restrictions, etc.",
        height=100,
        help="Add any specific requests or requirements for your trip"
    )
    
    # Generate final itinerary button
    if st.button("🎯 Generate My Personalized Itinerary", type="primary"):
        with st.spinner("🎯 Creating your personalized itinerary based on your preferences..."):
            # Use GitHub Models - Planner Agent
            planner_llm = ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key=github_token,
                openai_api_base="https://models.inference.ai.azure.com"
            )
            
            planner_input = f"""
CRITICAL: Respond ONLY in {trip_data['language']} language. Every word of the itinerary must be in {trip_data['language']}.

You are a senior travel planner. Based on the research results and user preferences, create a detailed day-wise itinerary.
Follow the planner instructions: Generate a detailed day-wise itinerary, quote facts from research results (do not fabricate), make it engaging and tailored to user's budget and language.

DESTINATION: {trip_data['destination']}
DURATION: {trip_data['num_days']} days
BUDGET: ₹{trip_data['budget']} INR total
LANGUAGE: {trip_data['language']} - IMPORTANT: Use this language for ALL content

RESEARCH RESULTS:
{st.session_state.research_results}

USER PREFERENCES AND SPECIFIC SELECTIONS:
- Accommodation Preference: {accommodation_pref}
- Selected Accommodation: {selected_accommodation}
- Activity Preference: {activity_pref}  
- Selected Activities: {selected_activities}
- Dining Preference: {dining_pref}
- Selected Dining Options: {selected_dining}
- Transportation Preference: {transport_pref}
- Selected Transportation: {selected_transport}
- Special Requests: {special_requests}

Please create a comprehensive itinerary that includes:
1. Day-by-day schedule with specific activities and timing based on user's exact selections
2. Use the specific accommodation option selected by the user
3. Include the exact restaurants/dining options the user chose
4. Use the selected transportation method throughout the trip
5. Estimated costs within the ₹{trip_data['budget']} INR total budget
6. Cultural tips and local customs
7. Emergency contacts and important information

Format the response clearly with day headers and detailed activities. Ensure all recommendations align with their specific selections and stay within the ₹{trip_data['budget']} INR budget. All prices should be in Indian Rupees (INR).

IMPORTANT: Structure the itinerary to specifically include the user's selected options rather than generic suggestions.
REMEMBER: Write EVERYTHING in {trip_data['language']}, not English or any other language.
"""
            
            itinerary_response = planner_llm.invoke(planner_input)
            itinerary_text = itinerary_response.content
            
        st.success("✅ Your personalized itinerary is ready!")
        st.subheader("🗓️ Your Complete Travel Itinerary")
        st.write(itinerary_text)

        # Save to session with all data including preferences and specific selections
        trip_id = str(uuid.uuid4())[:8]
        st.session_state.trip_history.append({
            "id": trip_id,
            "destination": trip_data['destination'],
            "days": trip_data['num_days'],
            "budget": trip_data['budget'],
            "language": trip_data['language'],
            "itinerary": itinerary_text,
            "research": st.session_state.research_results,
            "preferences": {
                "accommodation": accommodation_pref,
                "selected_accommodation": selected_accommodation,
                "activities": activity_pref,
                "selected_activities": selected_activities,
                "dining": dining_pref,
                "selected_dining": selected_dining,
                "transportation": transport_pref,
                "selected_transport": selected_transport,
                "special_requests": special_requests
            },
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

        # Export Options - PDF and Calendar Only
        st.subheader("📥 Export Your Itinerary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # PDF Export with multiple fallback methods
            if st.button("📄 Generate PDF", type="primary"):
                export_trip_id = str(uuid.uuid4())[:8]
                with st.spinner("📄 Generating PDF..."):
                    try:
                        # Create clean HTML for PDF
                        html_content = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <meta charset="UTF-8">
                            <title>Travel Itinerary - {trip_data['destination']}</title>
                            <style>
                                body {{ 
                                    font-family: Arial, sans-serif; 
                                    margin: 20px; 
                                    line-height: 1.6; 
                                    color: #333;
                                }}
                                .header {{ 
                                    text-align: center; 
                                    color: #2E86AB; 
                                    margin-bottom: 30px; 
                                    border-bottom: 2px solid #2E86AB;
                                    padding-bottom: 20px;
                                }}
                                .section {{ 
                                    margin: 20px 0; 
                                    padding: 15px;
                                    border-left: 4px solid #A23B72;
                                    background-color: #f9f9f9;
                                }}
                                .section-title {{ 
                                    color: #A23B72; 
                                    font-size: 18px; 
                                    font-weight: bold; 
                                    margin-bottom: 10px; 
                                }}
                                .selections {{
                                    background-color: #e8f4f8;
                                    padding: 15px;
                                    border-radius: 5px;
                                    margin: 10px 0;
                                }}
                            </style>
                        </head>
                        <body>
                            <div class="header">
                                <h1>🌍 Travel Itinerary: {trip_data['destination']}</h1>
                                <p><strong>Duration:</strong> {trip_data['num_days']} days | <strong>Budget:</strong> ₹{trip_data['budget']} INR | <strong>Language:</strong> {trip_data['language']}</p>
                                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                            </div>
                            
                            <div class="section">
                                <div class="section-title">🔍 Research Results</div>
                                <div>{st.session_state.research_results.replace(chr(10), '<br/>')}</div>
                            </div>
                            
                            <div class="section">
                                <div class="section-title">✨ Your Selections</div>
                                <div class="selections">
                                    <p><strong>🏨 Accommodation:</strong> {selected_accommodation} ({accommodation_pref})</p>
                                    <p><strong>🎯 Activities:</strong> {', '.join(selected_activities)} ({activity_pref})</p>
                                    <p><strong>🍽️ Dining:</strong> {', '.join(selected_dining)} ({dining_pref})</p>
                                    <p><strong>🚗 Transportation:</strong> {selected_transport} ({transport_pref})</p>
                                    {f'<p><strong>💭 Special Requests:</strong> {special_requests}</p>' if special_requests else ''}
                                </div>
                            </div>
                            
                            <div class="section">
                                <div class="section-title">🗓️ Detailed Itinerary</div>
                                <div>{itinerary_text.replace(chr(10), '<br/>')}</div>
                            </div>
                        </body>
                        </html>
                        """
                        
                        # Try multiple PDF generation methods
                        pdf_generated = False
                        pdf_filename = f"itinerary_{trip_data['destination'].replace(' ', '_')}_{export_trip_id}.pdf"
                        
                        # Method 1: Try weasyprint first (most reliable)
                        try:
                            from weasyprint import HTML
                            HTML(string=html_content).write_pdf(pdf_filename)
                            pdf_generated = True
                            st.info("✅ PDF generated using WeasyPrint")
                            
                        except Exception as weasy_error:
                            # Method 2: Try reportlab as fallback
                            try:
                                from reportlab.lib.pagesizes import letter
                                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                                from reportlab.lib.styles import getSampleStyleSheet
                                from reportlab.lib.units import inch
                                
                                doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
                                styles = getSampleStyleSheet()
                                story = []
                                
                                # Add content using reportlab
                                title = Paragraph(f"Travel Itinerary: {trip_data['destination']}", styles['Title'])
                                story.append(title)
                                story.append(Spacer(1, 12))
                                
                                # Trip details
                                details = f"Duration: {trip_data['num_days']} days | Budget: ₹{trip_data['budget']} INR | Language: {trip_data['language']}"
                                details_para = Paragraph(details, styles['Normal'])
                                story.append(details_para)
                                story.append(Spacer(1, 12))
                                
                                # Research section
                                research_title = Paragraph("Research Results", styles['Heading2'])
                                story.append(research_title)
                                research_para = Paragraph(st.session_state.research_results[:1000].replace('<', '&lt;').replace('>', '&gt;'), styles['Normal'])
                                story.append(research_para)
                                story.append(Spacer(1, 12))
                                
                                # Selections section
                                selections_title = Paragraph("Your Selections", styles['Heading2'])
                                story.append(selections_title)
                                selections_text = f"Accommodation: {selected_accommodation} | Activities: {', '.join(selected_activities)[:100]} | Dining: {', '.join(selected_dining)[:100]} | Transportation: {selected_transport}"
                                selections_para = Paragraph(selections_text.replace('<', '&lt;').replace('>', '&gt;'), styles['Normal'])
                                story.append(selections_para)
                                story.append(Spacer(1, 12))
                                
                                # Itinerary section
                                itinerary_title = Paragraph("Detailed Itinerary", styles['Heading2'])
                                story.append(itinerary_title)
                                itinerary_para = Paragraph(itinerary_text[:2000].replace('<', '&lt;').replace('>', '&gt;'), styles['Normal'])
                                story.append(itinerary_para)
                                
                                doc.build(story)
                                pdf_generated = True
                                st.info("✅ PDF generated using ReportLab")
                                
                            except Exception as reportlab_error:
                                st.error(f"❌ All PDF generation methods failed. Error: {str(reportlab_error)}")
                        
                        if pdf_generated:
                            with open(pdf_filename, "rb") as f:
                                pdf_data = f.read()
                                st.download_button(
                                    "📥 Download PDF", 
                                    pdf_data, 
                                    file_name=pdf_filename, 
                                    mime="application/pdf",
                                    key=f"pdf_download_{export_trip_id}"
                                )
                            
                            # Clean up
                            try:
                                os.remove(pdf_filename)
                            except:
                                pass
                            st.success("✅ PDF ready for download!")
                        else:
                            # Fallback: offer text download
                            formatted_content = f"""
=== TRAVEL ITINERARY ===
Destination: {trip_data['destination']}
Duration: {trip_data['num_days']} days
Budget: ₹{trip_data['budget']} INR
Language: {trip_data['language']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

=== RESEARCH RESULTS ===
{st.session_state.research_results}

=== YOUR SELECTIONS ===
🏨 Accommodation: {selected_accommodation} ({accommodation_pref})
🎯 Activities: {', '.join(selected_activities)} ({activity_pref})
🍽️ Dining: {', '.join(selected_dining)} ({dining_pref})
🚗 Transportation: {selected_transport} ({transport_pref})
{f'💭 Special Requests: {special_requests}' if special_requests else ''}

=== DETAILED ITINERARY ===
{itinerary_text}
"""
                            st.warning("⚠️ PDF generation unavailable. Offering text download instead.")
                            st.download_button(
                                "📄 Download as Text File", 
                                formatted_content, 
                                file_name=f"itinerary_{trip_data['destination'].replace(' ', '_')}_{export_trip_id}.txt", 
                                mime="text/plain",
                                key=f"fallback_text_download_{export_trip_id}"
                            )
                        
                    except Exception as pdf_error:
                        st.error(f"❌ Export failed: {str(pdf_error)}")
        
        with col2:
            # Calendar Export
            if st.button("📅 Download Calendar (.ics)", type="secondary"):
                export_trip_id = str(uuid.uuid4())[:8]
                with st.spinner("📅 Generating Calendar..."):
                    try:
                        from ics import Calendar, Event
                        
                        cal = Calendar()
                        start_date = datetime.now().date()
                        
                        # Create calendar events for each day
                        for i in range(trip_data['num_days']):
                            e = Event()
                            e.name = f"🌍 {trip_data['destination']} Trip - Day {i+1}"
                            e.begin = start_date + timedelta(days=i)
                            e.description = f"Day {i+1} of your {trip_data['destination']} trip\\n\\nBudget: ₹{trip_data['budget']} INR\\n\\nSelected Options:\\n- Accommodation: {selected_accommodation}\\n- Activities: {', '.join(selected_activities)}\\n- Dining: {', '.join(selected_dining)}\\n- Transport: {selected_transport}\\n\\nItinerary:\\n{itinerary_text[:300]}..."
                            e.duration = timedelta(hours=8)  # 8-hour events
                            cal.events.add(e)
                        
                        ics_filename = f"trip_{trip_data['destination'].replace(' ', '_')}_{export_trip_id}.ics"
                        ics_content = str(cal)
                        
                        st.download_button(
                            "📥 Download Calendar", 
                            ics_content, 
                            file_name=ics_filename, 
                            mime="text/calendar",
                            key=f"calendar_download_{export_trip_id}"
                        )
                        st.success("✅ Calendar file ready for download!")
                        
                    except Exception as e:
                        st.error(f"❌ Error creating calendar: {str(e)}")
                        st.info("💡 Installing required packages...")
                        # Attempt to install ics package
                        try:
                            import subprocess
                            subprocess.check_call(["pip", "install", "ics"])
                            st.info("📦 ics package installed. Please try again.")
                        except:
                            st.warning("⚠️ Could not auto-install calendar dependencies. You may need to install the 'ics' package manually.")

        # Enhanced Map with better geocoding
        st.subheader("🗺️ Interactive Destination Map")
        
        if not MAP_AVAILABLE:
            st.warning("⚠️ Interactive maps are not available. The required mapping libraries are not installed.")
            st.info(f"📍 Your destination: {trip_data['destination']}")
            st.info("🗺️ You can search for this destination on Google Maps for detailed location information.")
        else:
            # Simplified geocoding approach
            def get_coordinates(destination_name):
                """Get coordinates using a simple mapping approach with AI fallback"""
                # Common destinations mapping
                location_coords = {
                    "delhi": (28.6139, 77.2090),
                    "mumbai": (19.0760, 72.8777),
                    "bangalore": (12.9716, 77.5946),
                    "chennai": (13.0827, 80.2707),
                    "kolkata": (22.5726, 88.3639),
                    "hyderabad": (17.3850, 78.4867),
                    "pune": (18.5204, 73.8567),
                    "ahmedabad": (23.0225, 72.5714),
                    "jaipur": (26.9124, 75.7873),
                    "goa": (15.2993, 74.1240),
                    "kerala": (10.8505, 76.2711),
                    "kashmir": (34.0837, 74.7973),
                    "rajasthan": (27.0238, 74.2179),
                    "uttarakhand": (30.0668, 79.0193),
                    "himachal pradesh": (31.1048, 77.1734),
                    "paris": (48.8566, 2.3522),
                    "london": (51.5074, -0.1278),
                    "new york": (40.7128, -74.0060),
                    "tokyo": (35.6762, 139.6503),
                    "bangkok": (13.7563, 100.5018),
                    "dubai": (25.2048, 55.2708),
                    "singapore": (1.3521, 103.8198),
                    "malaysia": (4.2105, 101.9758),
                    "indonesia": (-0.7893, 113.9213),
                    "thailand": (15.8700, 100.9925),
                    "nepal": (28.3949, 84.1240),
                    "bhutan": (27.5142, 90.4336),
                    "sri lanka": (7.8731, 80.7718),
                    "maldives": (3.2028, 73.2207)
                }
                
                dest_lower = destination_name.lower()
                for key, coords in location_coords.items():
                    if key in dest_lower:
                        return coords
                
                # AI fallback for unknown destinations
                try:
                    geocode_llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        openai_api_key=github_token,
                        openai_api_base="https://models.inference.ai.azure.com"
                    )
                    
                    geocode_prompt = f"""
                    What are the latitude and longitude coordinates for {destination_name}?
                    Respond ONLY with: latitude,longitude
                    Example: 28.6139,77.2090
                    """
                    
                    response = geocode_llm.invoke(geocode_prompt)
                    coords_text = response.content.strip()
                    
                    if ',' in coords_text:
                        lat_str, lon_str = coords_text.split(',')
                        return float(lat_str.strip()), float(lon_str.strip())
                except:
                    pass
                    
                # Default fallback to India center
                return 20.5937, 78.9629
            
            try:
                with st.spinner("🌍 Locating destination on map..."):
                    lat, lon = get_coordinates(trip_data['destination'])
                    st.success(f"📍 Located: {trip_data['destination']} at {lat:.4f}, {lon:.4f}")
                    
                # Create and display map
                import folium
                from streamlit_folium import st_folium
                
                map_obj = folium.Map(
                    location=[lat, lon], 
                    zoom_start=8,
                    tiles='OpenStreetMap'
                )
                
                # Add destination marker
                folium.Marker(
                    [lat, lon], 
                    popup=f"🎯 {trip_data['destination']}",
                    tooltip=f"📍 {trip_data['destination']} - Your destination!",
                    icon=folium.Icon(color='red', icon='star')
                ).add_to(map_obj)
                
                # Display map
                map_data = st_folium(map_obj, width=700, height=400, returned_objects=["last_object_clicked"])
                st.caption(f"📍 Showing location of {trip_data['destination']}")
                
            except Exception as e:
                st.warning(f"⚠️ Map display error: {str(e)}")
                st.info(f"📍 Your destination: {trip_data['destination']}")
                lat, lon = get_coordinates(trip_data['destination'])
                st.info(f"🗺️ Coordinates: {lat:.4f}, {lon:.4f}")
                
                # Fallback: show a simple text-based location info
                st.markdown(f"""
                **📍 Location Information:**
                - **Destination:** {trip_data['destination']}
                - **Coordinates:** {lat:.4f}, {lon:.4f}
                - You can search for "{trip_data['destination']}" on Google Maps for detailed location information.
                """)

# Enhanced trip history with research data
if st.session_state.trip_history:
    st.markdown("## 📚 Trip History (Current Session)")
    st.markdown("*All your planned trips are saved in this session*")
    
    for i, trip in enumerate(st.session_state.trip_history[::-1]):  # Most recent first
        with st.expander(f"🌍 {trip['destination']} - {trip['days']} days ({trip['date']})", expanded=(i==0)):
            
            # Trip summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🎯 Destination", trip['destination'])
            with col2:
                st.metric("📅 Duration", f"{trip['days']} days")
            with col3:
                st.metric("💰 Budget", f"₹{trip['budget']} INR")
            with col4:
                st.metric("🗣️ Language", trip['language'])
            
            # Create tabs for better organization
            tab1, tab2, tab3 = st.tabs(["📋 Itinerary", "🔍 Research", "🎯 Preferences"])
            
            with tab1:
                st.markdown("### 🗓️ Day-by-Day Itinerary")
                st.write(trip["itinerary"])
                
            with tab2:
                st.markdown("### 🔍 Research Results")
                research_data = trip.get("research", "No research data available for this trip.")
                st.write(research_data)
                
            with tab3:
                st.markdown("### 🎯 User Preferences & Selections")
                preferences = trip.get("preferences", {})
                if preferences:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"🏨 **Accommodation:** {preferences.get('accommodation', 'N/A')}")
                        if preferences.get('selected_accommodation'):
                            st.write(f"   ✅ **Selected:** {preferences.get('selected_accommodation')}")
                        st.write(f"🍽️ **Dining:** {preferences.get('dining', 'N/A')}")
                        if preferences.get('selected_dining'):
                            st.write(f"   ✅ **Selected:** {', '.join(preferences.get('selected_dining', []))}")
                    with col2:
                        st.write(f"🎯 **Activities:** {preferences.get('activities', 'N/A')}")
                        if preferences.get('selected_activities'):
                            st.write(f"   ✅ **Selected:** {', '.join(preferences.get('selected_activities', []))}")
                        st.write(f"🚗 **Transportation:** {preferences.get('transportation', 'N/A')}")
                        if preferences.get('selected_transport'):
                            st.write(f"   ✅ **Selected:** {preferences.get('selected_transport')}")
                    if preferences.get('special_requests'):
                        st.write(f"✨ **Special Requests:** {preferences.get('special_requests')}")
                else:
                    st.write("No preference data available for this trip.")
