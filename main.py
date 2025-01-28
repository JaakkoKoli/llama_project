import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from taipy.gui import Gui, Markdown

page = Markdown("main.md", style="main.css")

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

category_string = """
Sports: Swimming, Skiing, Cycling, Fishing, Camping, Ball games\n
Electronics: Kitchen, Gaming\n
Toys: Board games, Dolls, Cars and Vehicles, LEGO, Summer toys, Other toys\n
Beauty: Makeup, Skincare, Nails, Hair, Body, Perfumes
"""

SYS_PROMPT_MAIN = """
Your job is to decide the type of action that would provide the greatest amount of value for the customer based on the customer\'s prompt. 
Answer only with the name of the action - nothing else. There are three different types of actions that can be taken that are the following:\n
\n
Name of the action: SEARCH \n
Description of the action: Help the user search for a product if the user mentioned any of the following search criteria: price, whether the product is on sale, brand of the product, one or more keywords for search, product category. For example the customer might ask for phones that are on sale. \n
\n
Name of the action: CATEGORY \n
Description of the action: Help the user find a product category that is most likely to contain what the user is looking for. For example the customer might ask where computers can be found. \n
\n
Name of the action: NOTHING \n
Description of the action: None of the other options apply, so nothing will be done.
"""

SYS_PROMPT_SEARCH = f"""
Your job is to figure out the best search options for the customer\'s prompt. 
The answer should be in the format \"option1:parameter1-parameter2, option2:parameter1\", so for example \"Price:0.0-55.0, Sale:true, Keyword:football\". Only separate different options with a comma, do not end the reply with one.\n
\n
The following are the available search options and the associated parameter(s). \n
\n
Price: Search for products within a specific price range. Parameters: minimum price, maximum price. Use 0 as the default minimum price if asked for cheaper than x, use 100000 as the default maximum if asked for more expensive than x. For example \"Price:10.0-150.0\".\n
Sale: Search for whether the product is on sale. Parameter: true/false based on if the product(s) should be on sale. For example \"Sale:true\".\n
Brand: Search for whether the product from a specific brand. Parameter: name of the brand. For example \"Brand:Dell\".\n
Keyword: Search for a product that includes the specific keyword(s). Parameter: list of keywords with each separated by ;. For example \"Keyword:football;shoe;small\".\n
Category: Search for a product in a specific category. Parameters: main category and secondary category as different parameters. For example \"Category:Sports-Fishing\".\n
\n
List of available categories in the form of \"main category: list of secondary categories\".
\n
{category_string}
"""

SYS_PROMPT_CATEGORY = f"""
Your job is to find the single most suitable primary and secondary product category pair for the customer\'s prompt. 
This could for example be the user looking for a specific product in which case choose the category that is the most likely to contain it. 
Only answer with the catagory names seperated by -. For example \"Sports - Cycling\" - the answer should be kept extremely short and simple, 
no need to answer with anything but the category pair. Do not add any extra information or reasoning to the answer, this is not needed.\n
\n
The categories should all be from the following list of primary and secondary categories in the form of (primary category):(list of secondary categories) \n
\n
{category_string}
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
accelerator = Accelerator()
model = AutoModelForCausalLM.from_pretrained(
    DEFAULT_MODEL,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    device_map=device,
)
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, use_safetensors=True)
model, tokenizer = accelerator.prepare(model, tokenizer)

customer_prompt = ""
answer = " "
style_answer = "no_answer"

def getCategory(answer):
    cats = answer.lower().replace("\n", "").replace(" - ", "-").split("-")
    res = {"main_category": "", "sub_category": "", "error": ""}
    main_category = ""
    categories = {"sports": ["swimming", "skiing", "cycling", "fishing", "camping", "ball games"],
              "electronics": ["kitchen", "gaming"],
              "toys": ["board games", "dolls", "cars and vehicles", "lego", "summer toys", "other toys"],
              "beauty": ["makeup", "skincare", "nails", "hair", "body", "perfumes"]}
    for cat in cats:
        if cat in categories:
            main_category = cat
            res["main_category"] = cat
    if main_category == "":
        res["error"] = "No matching product category found."
    for cat in cats:
        if cat in categories[main_category]:
            res["sub_category"] = cat
    return res

def getSearch(answer):
    search_option = answer.lower().replace(", ", ",").split(",")
    options = ["price", "sale", "brand", "keyword", "category"]
    search_option_dict = {}
    try:
        for s in search_option:
            s2 = s.split(":")
            if s2[0] in options:
                search_option_dict[s2[0]] = s2[1].split("-")
        res = ""
        # Edit the code below to do different things with the search parameters 
        for opt in search_option_dict.keys():
            res += f"{opt}: {search_option_dict[opt]}\n"
        return res
    except:
        return ""
    
def getLink(category_dict, link):
    if category_dict["error"] != "":
        return category_dict["error"]
    if category_dict["sub_category"] == "":
        return f"{link}/{category_dict["main_category"].replace(" ", "_")}"
    return f"{link}/{category_dict["main_category"].replace(" ", "_")}/{category_dict["sub_category"].replace(" ", "_")}"
    
def send_prompt(state):
    state.answer = "thinking..."
    state.style_answer = "thinking"
    conversation = [
        {"role": "system", "content": SYS_PROMPT_MAIN},
        {"role": "user", "content": state.customer_prompt},
    ]

    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=16
        )
    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = reply[reply.rfind("assistant")+len("assistant"):].replace(" ", "").replace("\n", "").upper()
    
    if reply == "CATEGORY":
        conversation = [
            {"role": "system", "content": SYS_PROMPT_CATEGORY},
            {"role": "user", "content": state.customer_prompt},
        ]

        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=64
            )
        reply = tokenizer.decode(output[0], skip_special_tokens=True)
        reply = reply[reply.rfind("assistant")+len("assistant"):].replace(" ", "").replace("\n", "").upper()
        
        state.answer = getLink(getCategory(reply), "https://examplesite.com")
        state.style_answer = ""
    elif reply == "SEARCH":
        conversation = [
            {"role": "system", "content": SYS_PROMPT_SEARCH},
            {"role": "user", "content": state.customer_prompt},
        ]

        prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=128
            )
        reply = tokenizer.decode(output[0], skip_special_tokens=True)
        reply = reply[reply.rfind("assistant")+len("assistant"):].replace(" ", "").replace("\n", "").upper()
        
        state.answer = getSearch(reply)
        state.style_answer = ""
    else:
        state.answer = ""
        state.style_answer = ""
    
if __name__ == "__main__":
    Gui(page=page).run(title="Chat bot")