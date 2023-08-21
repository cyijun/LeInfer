from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer,AutoModel
import torch
import json, datetime


embd_model = SentenceTransformer('/mnt/e/models/text2vec-large-chinese',device='cuda')

chat_tokenizer = AutoTokenizer.from_pretrained("/mnt/e/models/chatglm2-6b", trust_remote_code=True)
chat_model = AutoModel.from_pretrained("/mnt/e/models/chatglm2-6b", trust_remote_code=True).cuda()


DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def infer_chat(json_post_list):
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    response, history = chat_model.chat(chat_tokenizer,
                                    prompt,
                                    history=history,
                                    max_length=max_length if max_length else 2048,
                                    top_p=top_p if top_p else 0.7,
                                    temperature=temperature if temperature else 0.95)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)


def infer_embd(json_post_list):
    plaintext = json_post_list.get('plaintext')
    is_norm = json_post_list.get('is_norm')
    vector = embd_model.encode(plaintext)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": vector,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    return answer


def inference(json_post_raw):
    global chat_model, chat_tokenizer
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    model_type = json_post_list.get('type')
    if(model_type=='chat'):
        answer=infer_chat(json_post_list)
    elif(model_type=='embed'):
        answer=infer_embd(json_post_list)
    else:
        answer='unknown model type'

    torch_gc()
    return answer