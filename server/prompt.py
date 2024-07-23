QUERY_WITH_CONTEXT = """
Based on the following context:

{context}
______

Please answer this question using the above context: 

{question}

"""


QUERY_WITH_HISTORY = """
Based on the following context:
{context}
______
And considering our recent chat history:
{chat_history}
______

Please provide a succint yet enough information answer (max 200 words) to the following question:
{question}

___
"""


SYSTEM_ROLE = """
    You are an expertise in answering questions based on the provided context and the chat history with the user. 
    Ensure your responses are informative, concise, and directly address the user's queries.
"""


SUMMARIZE_CHAT = """
Summarize the following conversation into one sentence for each role. Make sure the summary is as comprehensive as possible.
Ensure your summary follows the exact format specified below:

{summary_prompt}
_____________

OUTPUT FORMAT IN JSON STRING
{{
  "user": "your-summarization-for-user-messages",
  "bot": "your-summarization-for-bot-messages"
}}
"""