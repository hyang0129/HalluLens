import time, subprocess

for attempt in range(20):
    time.sleep(60)

    # Check smoke test log
    r1 = subprocess.run(['tail', '-30', 'exploratory_work/method_1_pca/smoke_main_d128_t007_s050/training.log'],
                        capture_output=True, text=True, cwd='/mnt/home/hyang1/LLM_research/HalluLens')
    # Check main run log
    r2 = subprocess.run(['tail', '-30', 'exploratory_work/method_1_pca/main_d128_t007_s050/training.log'],
                        capture_output=True, text=True, cwd='/mnt/home/hyang1/LLM_research/HalluLens')
    # Check if processes still running
    r3 = subprocess.run(['pgrep', '-c', 'run_method1'], capture_output=True, text=True)

    print(f'\n=== Attempt {attempt+1} ({time.strftime("%H:%M:%S")}) ===', flush=True)
    print(f'run_method1 process count: {r3.stdout.strip()}', flush=True)
    print(f'--- smoke_main log ---\n{r1.stdout}', flush=True)
    print(f'--- main_d128 log ---\n{r2.stdout}', flush=True)

    # Stop early if we see training progress or completion in both
    smoke_done = 'results.json' in r1.stdout or 'Run complete' in r1.stdout or 'Epoch' in r1.stdout
    main_done = 'results.json' in r2.stdout or 'Run complete' in r2.stdout or 'Epoch' in r2.stdout
    if smoke_done and main_done:
        print('Both runs showing training progress! Stopping early.', flush=True)
        break
