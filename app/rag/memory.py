conversation_memory = {}


def get_history(conversation_id):
    return conversation_memory.get(
        conversation_id,
        []
    )


def add_message(
    conversation_id,
    role,
    content
):

    if conversation_id not in conversation_memory:
        conversation_memory[conversation_id] = []

    conversation_memory[conversation_id].append({
        "role": role,
        "content": content
    })


def clear_memory(conversation_id):

    if conversation_id in conversation_memory:
        del conversation_memory[conversation_id]