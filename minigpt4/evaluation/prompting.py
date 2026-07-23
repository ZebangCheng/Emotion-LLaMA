"""Prompt preparation shared by offline and training-time evaluation."""


def prepare_conversation_texts(texts, conversation_template):
    conversations = [conversation_template.copy() for _ in range(len(texts))]
    for conversation, text in zip(conversations, texts):
        conversation.append_message(conversation.roles[0], str(text))
        conversation.append_message(conversation.roles[1], None)
    return [conversation.get_prompt() for conversation in conversations]
