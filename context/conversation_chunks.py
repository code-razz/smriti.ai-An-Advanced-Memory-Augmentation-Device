conversation_chunks = [
    # === Conversation 1 ===
    {
        "text": """User: Did you bring the medicine?
Alex: Yes, I picked it up on the way.
User: Great, I need to take it after dinner.
Alex: Remember, the doctor said to avoid coffee.
User: Got it. I’ll skip the coffee tonight.
Alex: That’s good. Better safe than sorry.""",
        "metadata": {
            "conversation_id": "conv_20250705_01",
            "timestamp": "2025-07-05T19:00:00",
            "location": "Home",
            "participants": ["User", "Alex"],
            "tags": ["medicine", "pickup", "doctor"]
        }
    },
    {
        "text": """User: Is there any specific time to take it?
Alex: Ideally, after you've finished your meal, at least 30 minutes later.
User: Okay, I’ll take it after dinner then.
Alex: Great. Let me know if you need anything else.
User: Thanks again for picking it up.
Alex: No problem, happy to help!""",
        "metadata": {
            "conversation_id": "conv_20250705_01",
            "timestamp": "2025-07-05T19:03:00",
            "location": "Home",
            "participants": ["User", "Alex"],
            "tags": ["medicine", "timing", "thanks"]
        }
    },

    # === Conversation 2 ===
    {
        "text": """User: Let's buy groceries today.
Tom: Do we need milk?
User: Yes, and also eggs and bread.
Tom: I’ll make a list so we don’t forget anything.
User: We should also get some vegetables.
Tom: Good idea. I'll add them to the list.""",
        "metadata": {
            "conversation_id": "conv_20250705_02",
            "timestamp": "2025-07-05T17:20:00",
            "location": "Supermarket",
            "participants": ["User", "Tom"],
            "tags": ["shopping", "groceries", "list"]
        }
    },
    {
        "text": """User: Do we need any snacks?
Tom: Maybe some chips and nuts for the weekend.
User: I think we're good on snacks for now.
Tom: Alright, I'll grab everything from the list.
User: Let's head to checkout then.
Tom: Sure, let's get going.""",
        "metadata": {
            "conversation_id": "conv_20250705_02",
            "timestamp": "2025-07-05T17:23:30",
            "location": "Supermarket",
            "participants": ["User", "Tom"],
            "tags": ["groceries", "snacks", "checkout"]
        }
    },

    # === Conversation 3 ===
    {
        "text": """User: Did you see the game last night?
Sophia: Yes, it was intense!
User: I can't believe they won in the last minute!
Sophia: I know, I thought they were done for!
User: Who was your MVP?
Sophia: Definitely the quarterback. He pulled off some crazy moves.""",
        "metadata": {
            "conversation_id": "conv_20250705_03",
            "timestamp": "2025-07-05T20:00:00",
            "location": "Home",
            "participants": ["User", "Sophia"],
            "tags": ["sports", "game", "MVP"]
        }
    },
    {
        "text": """User: He was on fire! I hope they make it to the finals.
Sophia: Same here! It would be amazing to see them play in the championship.
User: I'm already looking forward to the next game.
Sophia: Me too, it's going to be another great one!
User: I’ll make some snacks for the next one!
Sophia: Sounds like a plan!""",
        "metadata": {
            "conversation_id": "conv_20250705_03",
            "timestamp": "2025-07-05T20:02:30",
            "location": "Home",
            "participants": ["User", "Sophia"],
            "tags": ["sports", "game", "snacks"]
        }
    },

    # === Conversation 4 ===
    {
        "text": """User: Do you want to watch a movie tonight?
Ethan: Sure! Any recommendations?
User: How about that new thriller on Netflix?
Ethan: Oh yeah, I heard it's really good!
User: It’s got some great reviews.
Ethan: Let’s do it then!""",
        "metadata": {
            "conversation_id": "conv_20250705_04",
            "timestamp": "2025-07-05T18:30:00",
            "location": "Home",
            "participants": ["User", "Ethan"],
            "tags": ["movie", "recommendation", "entertainment"]
        }
    },
    {
        "text": """User: I'll grab some popcorn.
Ethan: I'll get the drinks.
User: Movie night ready!
Ethan: Let’s get this show started!
User: Hope it’s as good as they say.
Ethan: Fingers crossed!""",
        "metadata": {
            "conversation_id": "conv_20250705_04",
            "timestamp": "2025-07-05T18:33:00",
            "location": "Home",
            "participants": ["User", "Ethan"],
            "tags": ["movie", "snacks", "hope"]
        }
    },

    # === Conversation 5 ===
    {
        "text": """User: Want to go for a walk later?
Jack: Sure, when do you want to go?
User: How about 6 PM?
Jack: Sounds good! Let’s meet at the park.
User: Perfect, I’ll be there on time.
Jack: See you then!""",
        "metadata": {
            "conversation_id": "conv_20250705_05",
            "timestamp": "2025-07-05T15:00:00",
            "location": "Park",
            "participants": ["User", "Jack"],
            "tags": ["outdoor", "walk"]
        }
    },
    {
        "text": """User: Do you want to take the long route or the short one?
Jack: I’m in the mood for a longer walk today.
User: Alright, let's do the long route then.
Jack: Great choice! I’ve been wanting to explore that trail again.
User: Let’s go!""",
        "metadata": {
            "conversation_id": "conv_20250705_05",
            "timestamp": "2025-07-05T15:10:00",
            "location": "Park",
            "participants": ["User", "Jack"],
            "tags": ["outdoor", "walk", "long route"]
        }
    },

    # === Conversation 6 ===
    {
        "text": """User: Did you hear about the new café opening downtown?
Olivia: Yes, I’ve heard the coffee is amazing!
User: I was thinking of checking it out tomorrow.
Olivia: Count me in! I’m always up for good coffee.
User: How about 10 AM?
Olivia: Perfect time! Let’s meet there.""",
        "metadata": {
            "conversation_id": "conv_20250705_06",
            "timestamp": "2025-07-05T14:00:00",
            "location": "Home",
            "participants": ["User", "Olivia"],
            "tags": ["coffee", "café", "plan"]
        }
    },
    {
        "text": """User: I’m going to try their caramel macchiato.
Olivia: I’ll go for the iced latte. It sounds refreshing.
User: I bet their pastries are good too.
Olivia: I’m definitely grabbing something sweet with my coffee!
User: Let’s go early to get a good spot.
Olivia: Agreed, let’s be there before it gets crowded!""",
        "metadata": {
            "conversation_id": "conv_20250705_06",
            "timestamp": "2025-07-05T14:05:00",
            "location": "Home",
            "participants": ["User", "Olivia"],
            "tags": ["coffee", "café", "plan", "pastries"]
        }
    }
]
