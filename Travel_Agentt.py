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
import io
import requests
from xml.sax.saxutils import escape as xml_escape
from dotenv import load_dotenv

# Optional imports with error handling
try:
    from ics import Calendar, Event
    ICS_AVAILABLE = True
except ImportError:
    ICS_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

load_dotenv()

# ---------------------------------------------------------
# Configuration for Groq
# ---------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

def create_llm(temperature=0.2):
    """Create a LangChain chat model client configured for Groq."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

    return ChatOpenAI(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        temperature=temperature,
    )

def clean_text_for_pdf(text):
    """Sanitize string for ReportLab XML parser and standard fonts."""
    if not text:
        return ""
    text = str(text).replace("₹", "INR ")
    text = xml_escape(text)
    return text.replace("\n", "<br/>")

def generate_pdf_bytes(trip_data, research_results, selections, itinerary_text):
    """Generate PDF binary in-memory without disk I/O errors."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title = Paragraph(f"Travel Itinerary: {clean_text_for_pdf(trip_data['destination'])}", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))

    details = (
        f"<b>Duration:</b> {trip_data['num_days']} days | "
        f"<b>Travelers:</b> {trip_data['num_people']} person(s) | "
        f"<b>Budget:</b> INR {trip_data['budget']}"
    )
    story.append(Paragraph(details, styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Research Results", styles['Heading2']))
    story.append(Paragraph(clean_text_for_pdf(research_results[:3000]), styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Your Selections", styles['Heading2']))
    sel_text = (
        f"Accommodation: {selections.get('accommodation')}<br/>"
        f"Activities: {', '.join(selections.get('activities', []))}<br/>"
        f"Dining: {', '.join(selections.get('dining', []))}<br/>"
        f"Transportation: {selections.get('transport')}"
    )
    story.append(Paragraph(clean_text_for_pdf(sel_text), styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Detailed Itinerary", styles['Heading2']))
    story.append(Paragraph(clean_text_for_pdf(itinerary_text), styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------------------------------------------------
# Streamlit UI & Session State Initialization
# ---------------------------------------------------------
st.set_page_config(page_title="✈️ AI Travel Planner", layout="centered")
st.title("AI Travel Planner ✈️ with Researcher & Planner Agents")
st.caption("Plan your trip with Groq, LangChain, and SerpAPI!")

if "trip_history" not in st.session_state:
    st.session_state.trip_history = []
if "research_results" not in st.session_state:
    st.session_state.research_results = None
if "current_trip_data" not in st.session_state:
    st.session_state.current_trip_data = None
if "show_preferences" not in st.session_state:
    st.session_state.show_preferences = False
if "current_itinerary" not in st.session_state:
    st.session_state.current_itinerary = None
if "current_selections" not in st.session_state:
    st.session_state.current_selections = {}

# --- Inputs ---
if not GROQ_API_KEY or not SERPAPI_API_KEY:
    st.warning("Please set your GROQ_API_KEY and SERPAPI_API_KEY in the .env file.")

destination = st.text_input("Where do you want to go?")
num_days = st.number_input("How many days do you want to travel for?", min_value=1, max_value=30, value=7)
budget = st.number_input("Budget (total amount in INR)", min_value=5000, max_value=5000000, value=100000, step=5000)
num_people = st.number_input("Number of People Traveling", min_value=1, max_value=20, value=2, step=1)

generate_clicked = st.button("🔍 Start Research & Planning")

if generate_clicked:
    if not (GROQ_API_KEY and SERPAPI_API_KEY):
        st.error("Please set your GROQ_API_KEY and SERPAPI_API_KEY in the .env file.")
    elif not destination:
        st.error("Please enter your travel destination.")
    else:
        try:
            llm = create_llm()

            search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
            search_tool = Tool(
                name="search_google",
                func=search.run,
                description="Searches for travel info related to destinations and activities."
            )
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            researcher_prompt = dedent(
                f"""
                You are a world-class travel researcher. Given a travel destination, duration, total budget, and number of people,
                generate search terms for relevant travel activities and accommodations suitable for {num_people} person(s).
                
                The group has a total ₹{budget} INR budget for {num_people} traveler(s) for {num_days} days.
                
                Generate 3 search terms related to the destination and days.
                For each, use `search_google` to fetch results and analyze.
                Return 10 most relevant insights aligned with the group size and budget.
                Maintain high quality and relevance.

                Present your findings in this format:

                **ACCOMMODATIONS OPTIONS:**
                - Option 1: [Name] - [Brief description] - [Price range in INR for {num_people} people]
                - Option 2: [Name] - [Brief description] - [Price range in INR for {num_people} people]
                - Option 3: [Name] - [Brief description] - [Price range in INR for {num_people} people]

                **ACTIVITY OPTIONS:**
                - Option 1: [Activity] - [Description] - [Duration/Time needed] - [Cost in INR for {num_people} people]
                - Option 2: [Activity] - [Description] - [Duration/Time needed] - [Cost in INR for {num_people} people]
                - Option 3: [Activity] - [Description] - [Duration/Time needed] - [Cost in INR for {num_people} people]

                **DINING OPTIONS:**
                - Option 1: [Restaurant/Food] - [Cuisine type] - [Price range in INR for {num_people} people]
                - Option 2: [Restaurant/Food] - [Cuisine type] - [Price range in INR for {num_people} people]
                - Option 3: [Restaurant/Food] - [Cuisine type] - [Price range in INR for {num_people} people]
                """
            )

            researcher = initialize_agent(
                tools=[search_tool],
                llm=llm,
                agent=AgentType.OPENAI_FUNCTIONS,
                verbose=False,
                memory=memory,
                max_iterations=3,
                handle_parsing_errors=True,
            )

            with st.spinner("🔍 Researching your destination..."):
                research_input = (
                    researcher_prompt
                    + f"\nDestination: {destination}\nDays: {num_days}\nBudget: {budget}\nTravelers: {num_people}"
                )
                research_results = researcher.invoke({"input": research_input})["output"]

                st.session_state.research_results = research_results
                st.session_state.current_trip_data = {
                    "destination": destination,
                    "num_days": num_days,
                    "budget": budget,
                    "num_people": num_people
                }
                st.session_state.show_preferences = True
                st.session_state.current_itinerary = None

            st.success("✓ Research completed! Now select your preferences below.")
            st.rerun()

        except Exception as error:
            st.error(f"❌ Research failed: {error}")
            st.info("Check your Groq API key, SerpAPI key, and model compatibility.")

# ---------------------------------------------------------
# Preferences Selection Section
# ---------------------------------------------------------
if st.session_state.show_preferences and st.session_state.research_results is not None:
    trip_data = st.session_state.current_trip_data

    try:
        llm = create_llm()
    except Exception as error:
        st.error(f"Unable to initialize Groq: {error}")
        st.stop()

    st.success("✓ Research completed - Now customize your preferences!")

    st.subheader("🔍 Research Results")
    st.write(st.session_state.research_results)

    st.subheader("🎯 Select Your Preferences")
    st.markdown("Choose your preferred options based on the research results above:")

    # Accommodation Selection
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

    if accommodation_pref != "No Preference":
        st.markdown("#### 🏨 **Top Accommodation Recommendations for Your Selection:**")

        with st.spinner("🔍 Finding specific accommodation options..."):
            accommodation_search_prompt = f"""
            Based on the research results and user preference for {accommodation_pref} in {trip_data['destination']}, 
            provide exactly 3 specific accommodation recommendations for {trip_data['num_people']} traveler(s) in this format:

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

        st.markdown("##### Choose your preferred accommodation:")
        selected_accommodation = st.selectbox(
            "Select your accommodation:",
            ["Option 1", "Option 2", "Option 3", "Let AI decide based on budget"],
            key="accommodation_selection",
            help="Choose the accommodation option that best fits your preferences"
        )
    else:
        selected_accommodation = "No Preference"

    # Activity Selection
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

    if activity_pref != "Mix of Everything":
        st.markdown("#### 🎯 **Top Activity Recommendations for Your Selection:**")

        with st.spinner("🔍 Finding specific activity options..."):
            activity_search_prompt = f"""
            Based on the research results and user preference for {activity_pref} in {trip_data['destination']}, 
            provide exactly 3 specific activity recommendations suitable for {trip_data['num_people']} traveler(s) in this format:

            **Option 1: [Activity Name]**
            - Description: [What to expect and details]
            - Cost: ₹[amount] total for {trip_data['num_people']} person(s)
            - Duration: [time needed]
            - Why recommended: [reason it fits preference]

            **Option 2: [Activity Name]**  
            - Description: [What to expect and details]
            - Cost: ₹[amount] total for {trip_data['num_people']} person(s)
            - Duration: [time needed]
            - Why recommended: [reason it fits preference]

            **Option 3: [Activity Name]**
            - Description: [What to expect and details]
            - Cost: ₹[amount] total for {trip_data['num_people']} person(s)
            - Duration: [time needed]
            - Why recommended: [reason it fits preference]

            Research context: {st.session_state.research_results}
            Budget: ₹{trip_data['budget']} INR total for {trip_data['num_days']} days
            """

            activity_recommendations = llm.invoke(activity_search_prompt).content
            st.write(activity_recommendations)

        st.markdown("##### Choose your preferred activities:")
        selected_activities = st.multiselect(
            "Select activities you want to include (you can choose multiple):",
            ["Option 1", "Option 2", "Option 3"],
            key="activity_selection",
            help="Choose one or more activities for your itinerary"
        )
        if not selected_activities:
            selected_activities = ["Option 1"]
    else:
        selected_activities = ["Mix of Everything"]

    # Dining Selection
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

    if dining_pref != "No Preference":
        st.markdown("#### 🍽️ **Top Dining Recommendations for Your Selection:**")

        with st.spinner("🔍 Finding specific dining options..."):
            dining_search_prompt = f"""
            Based on the research results and user preference for {dining_pref} in {trip_data['destination']}, 
            provide exactly 3 specific restaurant/dining recommendations suitable for {trip_data['num_people']} traveler(s) in this format:

            **Option 1: [Restaurant Name]**
            - Cuisine: [Type of cuisine]
            - Specialties: [Signature dishes]
            - Cost: ₹[amount] total for {trip_data['num_people']} person(s)
            - Location: [area/address]
            - Why recommended: [reason it fits preference]

            **Option 2: [Restaurant Name]**
            - Cuisine: [Type of cuisine]  
            - Specialties: [Signature dishes]
            - Cost: ₹[amount] total for {trip_data['num_people']} person(s)
            - Location: [area/address]
            - Why recommended: [reason it fits preference]

            **Option 3: [Restaurant Name]**
            - Cuisine: [Type of cuisine]
            - Specialties: [Signature dishes] 
            - Cost: ₹[amount] total for {trip_data['num_people']} person(s)
            - Location: [area/address]
            - Why recommended: [reason it fits preference]

            Research context: {st.session_state.research_results}
            Budget: ₹{trip_data['budget']} INR total for {trip_data['num_days']} days
            """

            dining_recommendations = llm.invoke(dining_search_prompt).content
            st.write(dining_recommendations)

        st.markdown("##### Choose your preferred dining options:")
        selected_dining = st.multiselect(
            "Select restaurants you want to include (you can choose multiple):",
            ["Option 1", "Option 2", "Option 3"],
            key="dining_selection",
            help="Choose one or more dining options for your itinerary"
        )
        if not selected_dining:
            selected_dining = ["Option 1"]
    else:
        selected_dining = ["No Preference"]

    # Transportation Selection
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

    if transport_pref != "No Preference":
        st.markdown("#### 🚗 **Transportation Recommendations for Your Selection:**")

        with st.spinner("🔍 Finding specific transportation options..."):
            transport_search_prompt = f"""
            Based on the research results and user preference for {transport_pref} in {trip_data['destination']}, 
            provide exactly 3 specific transportation recommendations for {trip_data['num_people']} traveler(s) in this format:

            **Option 1: [Service Name/Type]**
            - Description: [Service details and availability]
            - Cost: ₹[amount] total per day/trip for {trip_data['num_people']} person(s)
            - Coverage: [Areas served and convenience]
            - Booking: [How to book and tips]
            - Why recommended: [reason it fits preference]

            **Option 2: [Service Name/Type]**
            - Description: [Service details and availability]
            - Cost: ₹[amount] total per day/trip for {trip_data['num_people']} person(s)
            - Coverage: [Areas served and convenience]
            - Booking: [How to book and tips]
            - Why recommended: [reason it fits preference]

            **Option 3: [Service Name/Type]**
            - Description: [Service details and availability]
            - Cost: ₹[amount] total per day/trip for {trip_data['num_people']} person(s)
            - Coverage: [Areas served and convenience] 
            - Booking: [How to book and tips]
            - Why recommended: [reason it fits preference]

            Research context: {st.session_state.research_results}
            Budget: ₹{trip_data['budget']} INR total for {trip_data['num_days']} days
            """

            transport_recommendations = llm.invoke(transport_search_prompt).content
            st.write(transport_recommendations)

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
            planner_llm = create_llm()

            planner_input = f"""
You are a senior travel planner. Based on the research results and user preferences, create a detailed day-wise itinerary for {trip_data['num_people']} traveler(s).
Generate a detailed day-wise itinerary, quote facts from research results (do not fabricate), and make it engaging.

DESTINATION: {trip_data['destination']}
DURATION: {trip_data['num_days']} days
NUMBER OF TRAVELERS: {trip_data['num_people']}
BUDGET: ₹{trip_data['budget']} INR total

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

Format the response clearly with day headers and detailed activities. Ensure all recommendations align with their specific selections and stay within the ₹{trip_data['budget']} INR budget for all {trip_data['num_people']} travelers. All prices should be in Indian Rupees (INR).
"""

            itinerary_response = planner_llm.invoke(planner_input)
            st.session_state.current_itinerary = itinerary_response.content
            
            st.session_state.current_selections = {
                "accommodation": selected_accommodation, "accommodation_pref": accommodation_pref,
                "activities": selected_activities, "activity_pref": activity_pref,
                "dining": selected_dining, "dining_pref": dining_pref,
                "transport": selected_transport, "transport_pref": transport_pref,
                "special_requests": special_requests
            }

            trip_id = str(uuid.uuid4())[:8]
            st.session_state.trip_history.append({
                "id": trip_id,
                "destination": trip_data['destination'],
                "days": trip_data['num_days'],
                "budget": trip_data['budget'],
                "num_people": trip_data['num_people'],
                "itinerary": st.session_state.current_itinerary,
                "research": st.session_state.research_results,
                "preferences": st.session_state.current_selections,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            st.rerun()

# ---------------------------------------------------------
# Persistent Display: Itinerary and Exports
# ---------------------------------------------------------
if st.session_state.current_itinerary:
    trip_data = st.session_state.current_trip_data
    itinerary_text = st.session_state.current_itinerary
    selections = st.session_state.current_selections
    
    st.success("✅ Your personalized itinerary is ready!")
    st.subheader("🗓️ Your Complete Travel Itinerary")
    st.write(itinerary_text)

    # Export Options - PDF and Calendar
    st.subheader("📥 Export Your Itinerary")
    col1, col2 = st.columns(2)

    with col1:
        if REPORTLAB_AVAILABLE:
            export_trip_id = str(uuid.uuid4())[:8]
            pdf_data = generate_pdf_bytes(trip_data, st.session_state.research_results, selections, itinerary_text)
            st.download_button(
                "📄 Download PDF", 
                pdf_data, 
                file_name=f"itinerary_{trip_data['destination'].replace(' ', '_')}_{export_trip_id}.pdf", 
                mime="application/pdf"
            )
        else:
            st.warning("⚠️ PDF generation unavailable. Offering text download instead.")
            formatted_content = f"Destination: {trip_data['destination']}\nDuration: {trip_data['num_days']} days\nTravelers: {trip_data['num_people']}\n\n{itinerary_text}"
            st.download_button("📄 Download Text File", formatted_content, file_name=f"itinerary_{trip_data['destination']}.txt", mime="text/plain")

    with col2:
        if ICS_AVAILABLE:
            export_trip_id = str(uuid.uuid4())[:8]
            cal = Calendar()
            start_date = datetime.now().date()

            for i in range(trip_data['num_days']):
                e = Event()
                e.name = f"🌍 {trip_data['destination']} Trip - Day {i+1}"
                e.begin = start_date + timedelta(days=i)
                e.description = f"Day {i+1} of your {trip_data['destination']} trip\n\nBudget: ₹{trip_data['budget']} INR for {trip_data['num_people']} traveler(s)\n\nItinerary:\n{itinerary_text[:300]}..."
                e.duration = timedelta(hours=8)
                cal.events.add(e)

            ics_content = str(cal)
            st.download_button(
                "📥 Download Calendar", 
                ics_content, 
                file_name=f"trip_{trip_data['destination'].replace(' ', '_')}_{export_trip_id}.ics", 
                mime="text/calendar"
            )

# ---------------------------------------------------------
# Enhanced trip history
# ---------------------------------------------------------
if st.session_state.trip_history:
    st.markdown("---")
    st.markdown("## 📚 Trip History (Current Session)")
    st.markdown("*All your planned trips are saved in this session*")

    for i, trip in enumerate(st.session_state.trip_history[::-1]):
        with st.expander(f"🌍 {trip['destination']} - {trip['days']} days ({trip['date']})", expanded=(i==0)):

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Destination", trip['destination'])
            with col2:
                st.metric("📅 Duration", f"{trip['days']} days")
            with col3:
                st.metric("👥 Travelers", f"{trip.get('num_people', 1)} person(s)")

            tab1, tab2, tab3 = st.tabs(["📋 Itinerary", "🔍 Research", "🎯 Preferences"])

            with tab1:
                st.markdown("### 🗓️ Day-by-Day Itinerary")
                st.write(trip["itinerary"])

            with tab2:
                st.markdown("### 🔍 Research Results")
                st.write(trip.get("research", "No research data available for this trip."))

            with tab3:
                st.markdown("### 🎯 User Preferences & Selections")
                preferences = trip.get("preferences", {})
                if preferences:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"🏨 **Accommodation:** {preferences.get('accommodation_pref', 'N/A')}")
                        if preferences.get('accommodation'):
                            st.write(f"   ✅ **Selected:** {preferences.get('accommodation')}")
                        st.write(f"🍽️ **Dining:** {preferences.get('dining_pref', 'N/A')}")
                        if preferences.get('dining'):
                            st.write(f"   ✅ **Selected:** {', '.join(preferences.get('dining', []))}")
                    with c2:
                        st.write(f"🎯 **Activities:** {preferences.get('activity_pref', 'N/A')}")
                        if preferences.get('activities'):
                            st.write(f"   ✅ **Selected:** {', '.join(preferences.get('activities', []))}")
                        st.write(f"🚗 **Transportation:** {preferences.get('transport_pref', 'N/A')}")
                        if preferences.get('transport'):
                            st.write(f"   ✅ **Selected:** {preferences.get('transport')}")
                    if preferences.get('special_requests'):
                        st.write(f"✨ **Special Requests:** {preferences.get('special_requests')}")
                else:
                    st.write("No preference data available for this trip.")
