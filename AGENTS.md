# General rules to follow
1. Whenever you make substantial changes to the codebase that affects running, update REAME.md with the commands to run everything. 
2. Don't write private functions with one letter names. Function names must be human readable.
3. Whenever I ask you to write a report for experiments, it should contain: 
    - Hypothesis: what does the experiment even want to do? 
    - What did we do? Write down a config table if it involves training with the relevant hyperparams. 
    - What did we find? Write down our core findings, along with the numerical tables. Core findings must just be quick summaries.  
    - Next steps: what should we do next? discuss this with me before you write this. 

Throughout the report, be sure you use human readable english and don't have any filepaths or variable names unless i tell you to add them. Keep the reports to point and crisp, and make sure you're pulling the right numbers. It should also be a markdown file, unless I ask you to make it an artifact.

4. If there's training involved, maintain a CATALOGUE.md with the relevant paths for every experiment we did.
5. Don't reinvent the wheel, use what's available to you. for ex: if you're doing generations use either model.generate() first or do vllm if I ask. 
6. Document with comments to make it easier for me to go through the codebase. 
7. If I am asking you a question, answer it first and wait for me to signal you before going off and doing random things. 
8. Whenever you make edits to the codebase, do a quick edit pass to make sure that the core logic of the codebase isn't disturbed. 
9. When writing ML model architectures, be sure to check against a source (either the origin paper's codebase or an existing library) for consistency. 
10. When training, be sure to use wandb for logging the training metrics and any potential outputs. 
11. When showing me results from a training run, be sure to show me the actual numbers rather than your summaries. I need to look at the numbers myself.
