import ollama


def summarize_report(text):

    clean_text = text[:2000]

    prompt = f"""
    Summarize:
    - abnormalities
    - risks
    - recommendations

    Report:

    {clean_text}
    """

    response = ollama.chat(
        model='phi3',
        messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ]
    )

    return response['message']['content']