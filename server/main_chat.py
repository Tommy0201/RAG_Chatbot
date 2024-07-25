import os
import json
from dotenv import load_dotenv

import openai
from server.database_utils.retrieve_from_db import get_similarity_from_db
from server.database_utils.update_database import update_database

from server.prompt import QUERY_WITH_CONTEXT, SYSTEM_ROLE, QUERY_WITH_HISTORY, SUMMARIZE_CHAT

load_dotenv()
os.getenv("OPENAI_API_KEY")
chat_history = []

def generate_answer(user_message):
    #Chat history
    global chat_history
    
    #Get info from database as context
    context, references = get_similarity_from_db(user_message)
    reference_str = "".join(references)
    
    #Calling GPT API to get answer
    client = openai.Client()
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": QUERY_WITH_HISTORY.format(context=context,chat_history=chat_history,question=user_message)}
    ],
    # stream=True
    )
    bot_mess = completion.choices[0].message.content
    # ATTEMPT TO STREAM THE MESSAGE
    # stream_messages = []
    # bot_mess = ""
    # for idx,chunk in enumerate(completion):
    #     chunk_message = chunk.choices[0].delta.content
    #     if idx == 0:
    #         print(f"Bot: {chunk_message}",end='', flush=True)
    #     elif idx != 0 and chunk_message:
    #         print(chunk_message, end='', flush=True)
    #     stream_messages.append(chunk_message)
    # print(stream_messages)
    # for word in stream_messages:
    #     if word:
    #         bot_mess += word
    # print(bot_mess)

    #Chat history
    chat_history.append({"role":"user","content":user_message}) 
    chat_history.append({"role":"bot","content":bot_mess})
    chat_history = summarize_chat_history(chat_history) #Summarize chat history if it is too long

    return bot_mess, reference_str[:-2]

def summarize_chat_history(chat_history):
    history_length = sum(len(message["content"].split()) for message in chat_history)
    # print("chat_history before summarize: ",chat_history)

    if history_length > 300:
        
        #Formatting summary
        summary = """"""
        for message in chat_history:
            role = message["role"].capitalize()
            content = message["content"]
            summary += (f"""
                        {role}: {content}
                        """)
            
        #Calling GPT API
        client = openai.Client()
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": SUMMARIZE_CHAT.format(summary_prompt=summary)}
        ])
        summary = completion.choices[0].message.content
        
        #Processing summary
        summary_dict = json.loads(summary)
        chat_history = []
        if "user" in summary_dict:
            chat_history.append({"role": "user", "content": summary_dict["user"]})
        if "bot" in summary_dict:
            chat_history.append({"role": "bot", "content": summary_dict["bot"]})
        # print("new chat history: ",chat_history, "its type is: ", type(chat_history))
        
    return chat_history

def main():
    update_database() #Update database if files are added
    while True:
        user_input = input("User: ")
        if user_input in ["q","quit","exit"]:
            print("Bot: End of Convo!")
            break
        bot_message, references = generate_answer(user_input)
        print(f"Bot: {bot_message}")
        print(f"\nReferences: {references}")
    
    
# if __name__ == "__main__" :
#     main()

