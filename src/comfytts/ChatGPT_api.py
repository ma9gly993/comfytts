import httpx
from openai import OpenAI

class ChatGPT_api:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,  # True if you want the field to look like the one on the ClipTextEncode node
                        "default": "Write a very short story about comfy",
                    },
                ),
                "chatgpt_instruction_prompt": (
                    "STRING",
                    {
                        "multiline": True,  # True if you want the field to look like the one on the ClipTextEncode node
                        "default": "You are an AI assistant",
                    },
                ),

                "CHATGPT_API_KEY": (
                    "STRING",
                    {
                        "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                        "default": "Input your chatgpt api key here",
                    }
                )
            },

            "optional": {
                "PROXY": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "It can be empty. Proxy format: 'host:port:username:password.'",
                    }
                )

            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "main"
    CATEGORY = "comfytts"

    def main(self, prompt, chatgpt_instruction_prompt, PROXY, CHATGPT_API_KEY):
        timeout = 600
        connect = 5
        follow_redirects = True
        if PROXY:
            host, port, username, password = PROXY.split(':')
            proxy_url = f"socks5h://{username}:{password}@{host}:{port}"

            httpx_client = httpx.Client(proxy=proxy_url, timeout=timeout, follow_redirects=follow_redirects)
        else:
            httpx_client = httpx.Client(timeout=timeout, follow_redirects=follow_redirects)

        client = OpenAI(
            api_key=CHATGPT_API_KEY,
            http_client=httpx_client
        )

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": chatgpt_instruction_prompt},
                {"role": "user", "content": prompt}
            ]
        )


        return (completion.choices[0].message.content, )


