def call_azure_brewmaster_notes(
    beer_goal: str,
    hop_profile: dict,
    malt_profile: dict,
    yeast_profile: dict
) -> str:
    """
    Ask Azure OpenAI (your deployed model) for brewmaster-style guidance.
    Any errors from Azure get caught and turned into a friendly message
    so the Streamlit app doesn't crash.
    """

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")

    if not endpoint or not api_key or not deployment:
        return (
            "⚠️ Azure OpenAI credentials not found.\n\n"
            "Please set AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / "
            "AZURE_OPENAI_DEPLOYMENT in your Streamlit secrets."
        )

    # Build prompt for the model
    sys_prompt = (
        "You are an expert brewmaster. "
        "You are helping a craft brewer iterate on a recipe. "
        "You will get:\n"
        "1. The brewer's stated goal for this beer's character.\n"
        "2. The predicted hop aroma profile.\n"
        "3. The predicted malt / sweetness / body profile.\n"
        "4. The predicted yeast / fermentation profile.\n\n"
        "Please respond with concise, practical brewing advice:\n"
        "- High-level read on whether the beer hits the goal.\n"
        "- Hop adjustments (varieties, timing, amounts).\n"
        "- Malt/grist tweaks (which malts to add or reduce and why).\n"
        "- Fermentation guidance (yeast choice, temp, esters, mouthfeel).\n"
        "- Final summary for the brewer.\n\n"
        "Keep it under ~200 words. Use bullet points."
    )

    user_prompt = (
        f"Brewer's goal:\n{beer_goal}\n\n"
        f"Hops predicted profile:\n{repr(hop_profile)}\n\n"
        f"Malt predicted profile:\n{repr(malt_profile)}\n\n"
        f"Yeast predicted profile:\n{repr(yeast_profile)}\n\n"
        "Now provide the advice."
    )

    # Create AzureOpenAI client
    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        # IMPORTANT: this api_version must match what's supported in *your* region.
        # If you get consistent BadRequest, try checking the "Keys & Endpoint" page in Azure
        # and copy the API version they show there.
        api_version="2024-02-15-preview",
    )

    try:
        completion = client.chat.completions.create(
            model=deployment,  # in Azure this is your deployment name
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=400,
        )
    except Exception as e:
        # We swallow Azure errors and show them nicely to the user instead of crashing.
        return (
            "⚠️ I couldn't reach the Brewmaster AI just now.\n\n"
            "Details (for debugging):\n"
            f"{type(e).__name__}: {e}\n\n"
            "Things to check:\n"
            "- Azure deployment name in secrets (AZURE_OPENAI_DEPLOYMENT)\n"
            "- API version allowed in your Azure region\n"
            "- Content / safety filters\n"
            "- Quota / billing / region access to GPT-4.1-mini\n"
        )

    # Try to extract text safely
    ai_text = ""
    if getattr(completion, "choices", None):
        choice0 = completion.choices[0]
        # new SDK returns an object with choice0.message.content
        if hasattr(choice0, "message") and hasattr(choice0.message, "content"):
            ai_text = choice0.message.content or ""
        elif hasattr(choice0, "message") and isinstance(choice0.message, dict):
            ai_text = choice0.message.get("content", "") or ""

    if not ai_text:
        # fallback: dump entire completion
        ai_text = str(completion)

    return ai_text.strip() or "⚠️ Brewmaster AI returned an empty response."
