# AI Travel Agent ✈️

A powerful AI-powered travel itinerary planner built using **LangChain**, **OpenAI GPT-4o**, and **SerpAPI** all wrapped in a sleek **Streamlit** UI. This tool helps you research destinations, generate multi-day travel plans, visualize maps, export PDF/Calendar, and even track your trip history, all with natural language.

---

## Main Components

- **Researcher Agent**: Responsible for generating search terms based on the user's destination and travel duration, and searching the web for relevant activities and accommodations using SerpAPI.
- **Planner Agent**:  Uses the research results and user preferences to generates a day-wise engaging and budget-aware itinerary using GPT-4o.

---

## 🚀 Features

- Leverage LangChain to intelligently coordinate research and itinerary planning with dynamic prompt chaining and memory support.
- Research and discover exciting travel destinations, activities, and accommodations in real time using SerpAPI
- Personalized itinerary generation based on destination, duration, budget, and language
- Utilize the power of GPT-4o to generate intelligent and personalized travel plans
- Interactive destination map using OpenStreetMap (via Folium)
- Export your itinerary as a polished PDF 

---

### ⚙️How it works
- Step 1: User Input
Enter your travel destination, number of days, budget preference, language.

- Step 2: Intelligent Research
The Researcher agent uses LangChain with SerpAPI to fetch real-time information on activities, accommodations, and attractions based on your inputs.

- Step 3: Itinerary Generation
The Planner agent processes the research results using GPT-4o to create a detailed, day-wise, personalized travel itinerary.

- Step 4: Visualization & Customization
View your itinerary along with an interactive OpenStreetMap, adjust preferences if needed, and add budget and language filters.

- Step 5: Export & Sync
Download your itinerary as a PDF

- Step 6: Session History
Your recent trips are saved in the current session for easy review or re-generation without losing progress.

---

### 🗃️ Utilities
- **🔍 Real-time Google Search** (via SerpAPI)
- **🧠 Memory** (LangChain Conversation Memory)
- **📜 PDF Export** (via `pdfkit`)
- **🗺️ OpenStreetMap Integration** (via `folium`)
- **🗣️ Multilingual Support**
- **💸 Budget-Aware Suggestions**
- **📝 Trip History Management** (session-based)

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirement.txt
```

---

## 🧑‍💻 How to Run

```bash
streamlit run Travel_Agent.py
```

---

## 🔑 API Keys Required

- **OpenAI API Key** – get it from: https://platform.openai.com/account/api-keys or use Github Tocken
- **SerpAPI Key** – get it from: https://serpapi.com/users/sign_up

Both are free-tier friendly for limited use.

---

## 🛠️ Tech Stack
- LangChain
- OpenAI GPT-4o
- SerpAPI
- Streamlit
- PDFKit
- Folium + OpenStreetMap

---

## 📌 Coming Soon
- Saving trips to Firebase
- User authentication and chat history

---

## 🙌 Contributions & License
MIT License © 2025 
