import chainlit as cl
from pathlib import Path
import asyncio

from lyrical_homie import LyricalEngine

@cl.on_chat_start
async def start_chat():
    msg = cl.Message(content="Wbijam na rejon, ładuję bity i rymy...")
    await msg.send()

    db_path = Path(__file__).parent / "vector_db"
    engine = await asyncio.to_thread(LyricalEngine, db_path)

    cl.user_session.set("engine", engine)

    msg.content = "Elo ziom! Lyrical Homie na majku. Co Ci leży na serduchu?"
    await msg.update()


@cl.on_message
async def main(message: cl.Message):
    engine = cl.user_session.get("engine")
    u_msg = message.content

    async with cl.Step(name="Lyrical Homie...") as step:

        step.output = "Skanuję intencje..."
        await step.update()
        intent = await asyncio.to_thread(engine.classify_intent, u_msg)

        step.output = f"Wyciągam główny temat..."
        await step.update()
        target_tag = await asyncio.to_thread(engine.extract_search_tag, u_msg)

        step.output = f"Składam jako {target_tag}..."
        hyde_text = await asyncio.to_thread(engine.generate_hyde_answer, u_msg, intent, target_tag)

        step.output = "Przeszukuję archiwum rapu..."
        top_candidates = await asyncio.to_thread(engine.get_context, u_msg, hyde_text, intent, target_tag)

    if top_candidates == "Brak wyników.":
        reply = "Nie masz dla mnie cashu - nie zawracaj mi gitary"
    else:
        raw_reply = await asyncio.to_thread(engine.final_response, u_msg, top_candidates)
        cleaned_lines = [
            line.strip().replace(">", "").replace("*", "").replace('"', '')
            for line in raw_reply.split('\n') if line.strip()
        ]
        reply = "\\\n".join(cleaned_lines)

    await cl.Message(content=reply).send()