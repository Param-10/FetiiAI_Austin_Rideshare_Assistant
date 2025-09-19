## FetiiAI Austin Rideshare Assistant

### What We Built

We built Riley, a conversational AI assistant that knows everything about Fetii's rideshare data in Austin, Texas. Think ChatGPT but for Austin transportation patterns. Riley can answer questions about where people go, when they travel, and what the data reveals about Austin's group transportation scene.

The assistant processes over 2,000 recent rideshare trips and provides insights through natural conversation. Instead of charts and graphs, Riley talks you through the data with specific examples, interesting facts, and follow up questions that keep the conversation flowing.

Key features include Riley's personality system that remembers what you talked about, adapts to your communication style, and avoids repeating information. Riley can handle anything from casual responses like "sheesh" to specific questions about weekend pickup patterns or popular destinations.

### How It Works

The system runs on Python with several key components working together.

**Streamlit** powers the web interface and chat experience. Users type questions and get conversational responses in real time.

**Pandas** handles all the data processing. The system loads rideshare trip data and calculates distances, costs, peak hours, popular routes, and demographic breakdowns. It can work with Excel files or CSV backups automatically.

**Google Gemini 2.0 Flash** provides the natural language understanding and response generation. We built a smart intent classification system that recognizes different types of questions, from location queries to casual conversation.

**Custom Analytics Engine** processes user questions by filtering the data based on what they're asking about. It finds relevant patterns, calculates statistics, and identifies interesting insights like popular short routes perfect for scooters.

**Memory System** tracks conversation history so Riley remembers what you've discussed and won't repeat the same statistics. It also learns your interests and suggests related topics.

The AI uses advanced prompt engineering to maintain Riley's personality while providing accurate, specific insights from the actual data. Riley can match your energy level whether you're being casual or asking detailed questions.

### What I'd Improve or Add Next

Several areas could make Riley even better.

**Enhanced Data Sources** would let Riley answer more specific questions. Adding real time data, weather information, or event schedules could provide richer insights about why certain patterns happen.

**Deeper Personality** could include more Austin local knowledge. Riley could reference specific neighborhoods, events like SXSW or ACL, or local culture to make conversations feel more authentic.

**Advanced Analytics** could include route optimization suggestions, demand prediction, or personalized recommendations based on your travel patterns and preferences.

**Multi Session Memory** would let Riley remember you across different visits to the website. This could include your preferences, past questions, and ongoing interests.

**Voice Interface** could make the experience more natural. Speaking with Riley instead of typing could feel more like having a conversation with a knowledgeable local friend.

**Integration Features** could connect Riley to mapping services, calendar apps, or transportation booking systems to move from insights to action.
