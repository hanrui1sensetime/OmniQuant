from mlc_chat import ChatModule, ChatConfig, ConvConfig
from mlc_chat.callback import StreamToStdout
import os
import json_lines

datadir = '13b_llama_data_med'
resultdir = 'med_data_predict_lwc_let_debug_train512_predict8192'
partial_chat_config = ChatConfig(conv_config=ConvConfig(
    stop_tokens=[0],
    system=
    'Instructions: You are PULSE, a large language model trained by OpenMedLab. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-06-28',
    separator_style=0,
    seps=["</s>", "</s>"],
    stop_str="</s>",
    role_msg_sep=": ",
    role_empty_sep=": ",
    roles=['User', 'Helper'],
    add_bos=True))
cm = ChatModule(
    model="dist/omni_quant_lwc_let_w4a16g128_25-lwc_let_w4a16g128asym/params",
    lib_path=
    "dist/omni_quant_lwc_let_w4a16g128_25-lwc_let_w4a16g128asym/omni_quant_lwc_let_w4a16g128_25-lwc_let_w4a16g128asym-cuda.so",
    chat_config=partial_chat_config)

# prompt = "Instructions: You are PULSE, a large language model trained by OpenMedLab. Answer as concisely as possible.\nKnowledge cutoff: 2021-09-01\nCurrent date: 2023-06-28</s> User: "
prompt = ""  # "User: "  # 'User: '  # 'User: '
answer_start = ""  # "答: "  # "答: "
question_list = []
ans_list = []
if not os.path.exists(resultdir):
    os.mkdir(resultdir)

for file in os.listdir(datadir):
    question_list = []
    ans_list = []
    with open(os.path.join(datadir, file)) as f:
        for line in json_lines.reader(f):

            question = prompt + line['question'] + answer_start
            cm.reset_chat(partial_chat_config)
            output = cm.generate(
                prompt=question,
                progress_callback=StreamToStdout(callback_interval=3),
            )
            output = output.strip()
            if output.startswith('Helper: '):
                output = output[8:]
            '''
            output = output.strip().split("\n\n")[1:]
            output = "\n\n".join(output)

            start_pos = 0
            while start_pos < len(line['question']) and line['question'][start_pos] == output[start_pos]:
                start_pos += 1
            while start_pos < len(output) and output[start_pos:start_pos + 8] == "Helper: ":
                start_pos += 8
            '''
            question_list.append(line)
            ans_list.append(output)
            print(f"\nStatistics: {cm.stats()}\n")
    with open(os.path.join(resultdir, file), "w+", encoding="utf8") as f:
        import json
        for test_dataset_item, predict_output_item in zip(
                question_list, ans_list):
            f.write(
                json.dumps(
                    {
                        "type": test_dataset_item["type"],
                        "question": test_dataset_item["question"],
                        "reference_answer":
                        test_dataset_item["reference_answer"],
                        "predict_answer": predict_output_item.strip(),
                    },
                    ensure_ascii=False) + "\n")
