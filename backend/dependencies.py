graph = None
chat_model = None


def load_agent_resources():
    global graph, chat_model
    if graph is None or chat_model is None:
        from agent.graph import build_graph, chat_model as loaded_chat_model

        chat_model = loaded_chat_model
        if graph is None:
            graph = build_graph()
    return graph, chat_model
