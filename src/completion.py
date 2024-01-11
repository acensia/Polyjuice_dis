from enum import Enum
from dataclasses import dataclass
import openai
from openai import AsyncOpenAI

from src.moderation import moderate_message
from typing import Optional, List
from src.constants import (
    BOT_INSTRUCTIONS,
    BOT_NAME,
    EXAMPLE_CONVOS,
)
import discord
from src.base import Message, Prompt, Conversation, ThreadConfig
from src.utils import split_into_shorter_messages, close_thread, logger
from src.moderation import (
    send_moderation_flagged_message,
    send_moderation_blocked_message,
)

MY_BOT_NAME = BOT_NAME
MY_BOT_EXAMPLE_CONVOS = EXAMPLE_CONVOS
class changed:
    BOT_INSTRUCTIONS_MODIFIED = BOT_INSTRUCTIONS

class CompletionResult(Enum):
    OK = 0
    TOO_LONG = 1
    INVALID_REQUEST = 2
    OTHER_ERROR = 3
    MODERATION_FLAGGED = 4
    MODERATION_BLOCKED = 5


@dataclass
class CompletionData:
    status: CompletionResult
    reply_text: Optional[str]
    status_text: Optional[str]


client = AsyncOpenAI()


async def generate_morphing_inst(
        messages: str, user:str, thread_config:ThreadConfig
)->CompletionData:
    try:
        msg = f"You are prompt engineer to have gpt make responses better. \
        Give me an instruction to prompt gpt to respond by mimicking {messages}'s \
        characteristics, including personalities, experiences, speeches, and knowledges. Only the instruction sentence, without any explanation"

        rendered = [{"role" : "system", "content":msg}]

        print("instruction generating starts")
        res = await client.chat.completions.create(
            model=thread_config.model,
            messages=rendered,
            temperature=thread_config.temperature,
            top_p=1.0,
            max_tokens=thread_config.max_tokens,
            stop=["<|endoftext|>"],
        )
        print(res.choices[0].message.content)
        changed.BOT_INSTRUCTIONS_MODIFIED = res.choices[0].message.content + BOT_INSTRUCTIONS

        return CompletionData(
            status=CompletionResult.OK, reply_text=messages, status_text=None
        )
    except Exception as e:
        print(e)
        print("Error occurs")
        pass
        



async def generate_completion_response(
    messages: List[Message], user: str, thread_config: ThreadConfig
) -> CompletionData:
    try:
        print(changed.BOT_INSTRUCTIONS_MODIFIED)
        prompt = Prompt(
            header=Message(
                "system", f"Instructions for {MY_BOT_NAME}: {changed.BOT_INSTRUCTIONS_MODIFIED}"
            ),
            examples=MY_BOT_EXAMPLE_CONVOS,
            convo=Conversation(messages),
        )
        rendered = prompt.full_render(MY_BOT_NAME)
        response = await client.chat.completions.create(
            model=thread_config.model,
            messages=rendered,
            temperature=thread_config.temperature,
            top_p=1.0,
            max_tokens=thread_config.max_tokens,
            stop=["<|endoftext|>"],
        )
        reply = response.choices[0].message.content.strip()
        print(response.choices[0].message.content)
        if reply:
            flagged_str, blocked_str = moderate_message(
                message=(rendered[-1]["content"] + reply)[-500:], user=user
            )
            if len(blocked_str) > 0:
                return CompletionData(
                    status=CompletionResult.MODERATION_BLOCKED,
                    reply_text=reply,
                    status_text=f"from_response:{blocked_str}",
                )

            if len(flagged_str) > 0:
                return CompletionData(
                    status=CompletionResult.MODERATION_FLAGGED,
                    reply_text=reply,
                    status_text=f"from_response:{flagged_str}",
                )

        return CompletionData(
            status=CompletionResult.OK, reply_text=reply, status_text=None
        )
    except openai.BadRequestError as e:
        if "This model's maximum context length" in str(e):
            return CompletionData(
                status=CompletionResult.TOO_LONG, reply_text=None, status_text=str(e)
            )
        else:
            logger.exception(e)
            return CompletionData(
                status=CompletionResult.INVALID_REQUEST,
                reply_text=None,
                status_text=str(e),
            )
    except Exception as e:
        logger.exception(e)
        return CompletionData(
            status=CompletionResult.OTHER_ERROR, reply_text=None, status_text=str(e)
        )


async def process_response(
    user: str, thread: discord.Thread, response_data: CompletionData
):
    status = response_data.status
    reply_text = response_data.reply_text
    status_text = response_data.status_text
    if status is CompletionResult.OK or status is CompletionResult.MODERATION_FLAGGED:
        sent_message = None
        if not reply_text:
            sent_message = await thread.send(
                embed=discord.Embed(
                    description=f"**Invalid response** - empty response",
                    color=discord.Color.yellow(),
                )
            )
        else:
            shorter_response = split_into_shorter_messages(reply_text)
            for r in shorter_response:
                sent_message = await thread.send(r)
        if status is CompletionResult.MODERATION_FLAGGED:
            await send_moderation_flagged_message(
                guild=thread.guild,
                user=user,
                flagged_str=status_text,
                message=reply_text,
                url=sent_message.jump_url if sent_message else "no url",
            )

            await thread.send(
                embed=discord.Embed(
                    description=f"⚠️ **This conversation has been flagged by moderation.**",
                    color=discord.Color.yellow(),
                )
            )
    elif status is CompletionResult.MODERATION_BLOCKED:
        await send_moderation_blocked_message(
            guild=thread.guild,
            user=user,
            blocked_str=status_text,
            message=reply_text,
        )

        await thread.send(
            embed=discord.Embed(
                description=f"❌ **The response has been blocked by moderation.**",
                color=discord.Color.red(),
            )
        )
    elif status is CompletionResult.TOO_LONG:
        await close_thread(thread)
    elif status is CompletionResult.INVALID_REQUEST:
        await thread.send(
            embed=discord.Embed(
                description=f"**Invalid request** - {status_text}",
                color=discord.Color.yellow(),
            )
        )
    else:
        await thread.send(
            embed=discord.Embed(
                description=f"**Error** - {status_text}",
                color=discord.Color.yellow(),
            )
        )
