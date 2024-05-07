seeds="1 2 3"

for s in $seeds; do
    python3 question_form.py sequence --all | python3 produce_prompts.py --seed $s | python3 query.py -n openbuddy-llama2-13b-bf16 -m OpenBuddy/openbuddy-llama2-13b-v11.1-bf16 --exp_dir "trial"$s --index
    python3 question_form.py sequence --all | python3 produce_prompts.py --seed $s | python3 query.py -n chinese-alpaca-2-13b -m hfl/chinese-alpaca-2-13b --exp_dir "trial"$s --index
    python3 question_form.py sequence --all | python3 produce_prompts.py --seed $s | python3 query.py -n bloomz-7b1 -m bigscience/bloomz-7b1 --exp_dir "trial"$s --index
    python3 question_form.py sequence --all | python3 produce_prompts.py --seed $s | python3 query.py -n CausalLM-14B -m CausalLM/14B --exp_dir "trial"$s --index
done