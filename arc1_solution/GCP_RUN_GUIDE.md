# Running the ARC1 Solution (Tiny Recursion Model) on GCP

This guide explains how to run the ARC-AGI-1 training using Google Cloud's notebook environments (Vertex AI and Colab) and how to connect them to your local IDE.

## The Cost vs. Power Dilemma
The official ARC-AGI-1 training expects **4x H100 GPUs** and takes about **3 days**. 
- **Colab Pro/Pro+**: Very cheap (flat monthly fee), but **limited to 1 GPU** (max 1x A100). Training will take 4x longer (~12 days) and sessions time out every 12-24 hours, meaning you MUST implement checkpoint saving/resuming.
- **Vertex AI Workbench**: Very easy to set up and allows exactly **4x A100s/H100s**. However, it runs on Compute Engine under the hood, so the **hourly cost is the same as a raw VM**. The benefit is better UI, easier management, and auto-shutdown features.

---

## Option 1: Vertex AI Workbench (Easiest for 4x A100s/H100s)

Vertex AI Workbench gives you a managed Jupyter environment with underlying Compute Engine power.

1. Go to **Google Cloud Console > Vertex AI > Workbench**.
2. Click **Create New > User-Managed Notebook**.
3. Under **Compute**, select your GPUs (e.g., 4x NVIDIA A100).
4. **Environment**: Choose a PyTorch environment.
5. **Idle Shutdown**: Set this to 1-2 hours so you don't get billed if you forget to turn it off!
6. Click **Create**.

### Connecting to your local IDE (Antigravity/VS Code)
Because Vertex AI Workbench is essentially a Compute Engine VM, you can connect your local IDE directly to it via SSH.

1. Install the Google Cloud CLI (`gcloud`) on your local Mac.
2. Authenticate: `gcloud auth login`
3. Get the SSH command for your notebook:
   ```bash
   gcloud compute ssh --project [PROJECT_ID] --zone [ZONE] [NOTEBOOK_NAME]
   ```
4. In your Antigravity IDE (or VS Code), install the **Remote - SSH** extension.
5. Open your `~/.ssh/config` file and add the GCP host configurations. (Running the `gcloud compute config-ssh` command will automatically add your GCP VMs to your SSH config).
6. In your IDE, click the Remote SSH icon and select your Vertex AI notebook. Your IDE is now fully linked to the 4x A100s!

---

## Option 2: Google Colab Pro / Pro+ (Cheapest, but 1 GPU only)

If Vertex AI/Compute Engine is too costly, Colab Pro/Pro+ is the cheapest way to access an A100. 

1. Subscribe to Colab Pro or Pro+.
2. Create a new notebook and go to **Runtime > Change runtime type**.
3. Select **A100 GPU** (if available) and **High-RAM**.

### Connecting Colab to your local IDE
You can tunnel into a Google Colab instance using SSH, tricking your IDE into thinking it's a standard remote server.

1. In your Colab Notebook, install `colab-ssh` and `ngrok` (or Cloudflare Tunnels):
   ```python
   !pip install colab_ssh --upgrade
   from colab_ssh import launch_ssh_cloudflared
   # Set a password for the SSH connection
   launch_ssh_cloudflared(password="my_secure_password")
   ```
2. The notebook output will provide an SSH command (e.g., `ssh -p 22 root@some-cloudflare-url.trycloudflare.com`).
3. In your local IDE, use the **Remote - SSH** extension to connect using the provided host and password.
4. *Warning*: Colab will disconnect you after 12-24 hours. You must edit the `pretrain.py` script to save checkpoints to Google Drive, and resume from those checkpoints when you restart the Colab session.

---

## Step 3: Run the Training Process via IDE Terminal

Once your IDE is connected to either Vertex AI or Colab via Remote-SSH, open the IDE terminal. It is now executing on the remote GPU machine.

```bash
# Clone the repository
# git clone <your-repo-url>
# cd TinyRecursiveModels-main

# Install dependencies
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
pip install -r requirements.txt
pip install --no-cache-dir --no-build-isolation adam-atan2 

# Prepare the dataset
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation
```

**For Vertex AI (4 GPUs):**
```bash
tmux new -s arc1_training
run_name="pretrain_att_arc1concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm data_paths="[data/arc1concept-aug-1000]" arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 +run_name=${run_name} ema=True
```

**For Colab (1 GPU):**
*Note: You must drop `--nproc-per-node` to 1, or remove `torchrun` and run `pretrain.py` directly if it supports single GPU.*
```bash
run_name="pretrain_att_arc1concept_1gpu"
python pretrain.py \
arch=trm data_paths="[data/arc1concept-aug-1000]" arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 +run_name=${run_name} ema=True
```
