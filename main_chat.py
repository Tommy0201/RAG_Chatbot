import os
from dotenv import load_dotenv

import openai
from retrieve_from_db import get_similarity_from_db
from database import update_database

from prompt import QUERY_WITH_CONTEXT, SYSTEM_ROLE

load_dotenv()
os.getenv("OPENAI_API_KEY")

def generate_answer(user_message):
    context, references = get_similarity_from_db(user_message)
    reference_str = "".join(references)

    client = openai.Client()
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": QUERY_WITH_CONTEXT.format(context=context,question=user_message)}
    ]
    )
    bot_mess = completion.choices[0].message.content
    

    return bot_mess, reference_str[:-2]


if __name__ == "__main__" :
    update_database()
    while True:
        user_input = input("User: ")
        if user_input in ["q","quit","exit"]:
            print("Bot: End of Convo!")
            break
        bot_message, references = generate_answer(user_input)
        print(f"Bot: {bot_message}")
        print(f"References: {references}")
    
