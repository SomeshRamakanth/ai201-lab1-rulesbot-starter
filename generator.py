from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL

_client = Groq(api_key=GROQ_API_KEY)


def generate_response(query, retrieved_chunks):
    """
    Generate a grounded answer from retrieved rule chunks.

    TODO — Milestone 3:

    `retrieved_chunks` is the list returned by retrieve(). Each item is a dict:
      - "text"     : the chunk text
      - "game"     : the game name
      - "distance" : similarity score (you can use this to filter weak matches)

    Before writing code, talk through these with your group:
      - How will you format the chunks into a context block for the prompt?
      - What instructions will stop the model from answering beyond what the
        rules say? (Grounding is the whole point — a confident wrong answer
        is worse than an honest "I don't know.")
      - How will you surface which game each answer comes from?

    Your response should:
      1. Answer using only the retrieved context — not the model's general knowledge
      2. Make clear which game the answer comes from
      3. Say so clearly when the answer isn't in the loaded rules

    Return the response as a plain string.
    """
    if not retrieved_chunks:
        return (
            "I couldn't find anything relevant in the loaded rule books. "
            "Try rephrasing your question — or check that your ingestion pipeline is working."
        )

    # Optional: filter weak matches. Lower distance means stronger match.
    relevant_chunks = [
        chunk for chunk in retrieved_chunks
        if chunk.get("distance", 1.0) <= 0.65
    ]

    if not relevant_chunks:
        return (
            "I couldn't find a reliable answer in the loaded rule books. "
            "The retrieved passages were not relevant enough to answer confidently."
        )

    # Format retrieved chunks into context
    context = "RULE EXCERPTS:\n"
    for i, chunk in enumerate(relevant_chunks, start=1):
        context += (
            f"\n--- Excerpt {i} ---\n"
            f"Game: {chunk['game']}\n"
            f"Similarity distance: {chunk['distance']:.3f}\n"
            f"Rule text:\n{chunk['text']}\n"
        )

    # Build the system and user messages
    system_message = """You are RulesBot, a board game rules assistant.

You must answer using ONLY the rule excerpts provided by the user.

Rules:
1. Do not use outside knowledge.
2. Do not guess.
3. If the answer is not clearly present in the excerpts, say: "I don't see this in the loaded rules."
4. Always mention which game the answer comes from.
5. Keep the answer clear and concise.
6. If the question asks about one game but the retrieved excerpts are from a different game, say the loaded rules do not provide an answer for the requested game."""

    user_message = (
        f"{context}\n\n"
        f"QUESTION:\n{query}\n\n"
        "Answer using only the rule excerpts above."
    )
    
    try:
        completion = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=500,
        )

        return completion.choices[0].message.content.strip()

    except Exception as e:
        return f"Sorry, I ran into an error while generating the response: {e}"

    # Call Groq API
    completion = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_tokens=500,
    )

    return completion.choices[0].message.content.strip()
