from threading import Lock


# In-process conversation memory.
# The key is conversation_id, not video_id.
conversation_memory = {}

_memory_lock = Lock()

# Prevent memory from growing indefinitely.
MAX_MESSAGES_PER_CONVERSATION = 40


def get_history(conversation_id):
    """
    Return a copy of the conversation history.
    """
    if not conversation_id:
        return []

    with _memory_lock:
        history = conversation_memory.get(
            conversation_id,
            []
        )

        return [
            message.copy()
            for message in history
        ]


def add_message(
    conversation_id,
    role,
    content,
    video_id=None,
):
    """
    Add a message to a conversation.

    Supported roles:
    - user
    - assistant
    """

    if not conversation_id:
        return

    message = {
        "role": role,
        "content": content,
    }

    if video_id:
        message["video_id"] = video_id

    with _memory_lock:

        if conversation_id not in conversation_memory:
            conversation_memory[conversation_id] = []

        conversation_memory[conversation_id].append(
            message
        )

        # Keep only the most recent messages.
        if (
            len(conversation_memory[conversation_id])
            > MAX_MESSAGES_PER_CONVERSATION
        ):
            conversation_memory[conversation_id] = (
                conversation_memory[conversation_id][
                    -MAX_MESSAGES_PER_CONVERSATION:
                ]
            )


def clear_memory(conversation_id):
    """
    Clear the memory of a particular conversation.
    """

    if not conversation_id:
        return

    with _memory_lock:
        conversation_memory.pop(
            conversation_id,
            None
        )