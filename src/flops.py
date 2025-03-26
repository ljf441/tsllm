"""                                         SQUELETON OF QWEN 

Input X (Size: N)
       │
       ▼
  ┌──────────────────────────────────┐
  │ (1) Token Embeddings(151936, 896)│  <--- embed_tokens: Converts tokens into D-dimensional vectors (D=896, L=151936)
  └──────────────────────────────────┘
       │
       ▼
  ┌──────────────────────────────────┐
  │ (2) 24x Qwen2DecoderLayer        │  <--- Applies 24 Layers (K layers)
  └──────────────────────────────────┘
       │
       ▼
  ┌──────────────────────────────────────────────────────┐
  │ Inside each Qwen2DecoderLayer:                       │
  │  (a) input_layernorm (Qwen2RMSNorm)                  │  <--- Pre-normalization with RMSNorm
  │  (b) self_attn (Qwen2Attention)                      │
  │       - q_proj: Linear (D → D)    bias=True          │
  │       - k_proj: Linear (D → 128)  bias=True          │  <--- Simplifying this layer to a multi-headed Attention layer would be 14 heads (H) with D -> D Linear. 
  │       - v_proj: Linear (D → 128)  bias=True          │
  │       - o_proj: Linear (128 → D)  bias=False         │
  │  (c) post_attention_layernorm (Qwen2RMSNorm)         │  <--- Second normalization after attention
  │  (d) mlp (Qwen2MLP)                                  │
  │       - gate_proj: Linear (D → 4864) bias=False      │  <---- 4864 is the number of hidden units
  │       - up_proj: Linear (D → 4864)   bias=False      │
  │       - SiLU Activation Function                     │
  │       - down_proj: Linear (4864 → D) bias=False      │ 
  └──────────────────────────────────────────────────────┘
       │
       ▼
  ┌──────────────────────────────────┐
  │ (3) norm (Qwen2RMSNorm)          │  <--- Final normalization before projection
  └──────────────────────────────────┘
       │
       ▼
  ┌────────────────────────────────────────────────────────────────┐
  │ (4) Rotary Positional Embeddings (RoPE) (rotary_emb)           │  <--- positional encoding: They replace traditional 
  └────────────────────────────────────────────────────────────────┘       absolute positional embeddings with rotary positional information. (NO FLOPS)
       │
       ▼
  ┌────────────────────────────────────────────┐
  │ (5) lm_head (Linear Layer) bias=True       │  <--- Final projection to vocabulary (D → L)
  └────────────────────────────────────────────┘
    """


def total_flops(num_steps, batch_size, D, L, N, h_u, H, K, training=True):
    """
    Compute the total FLOPs for an experiment, accounting for number of epochs and batch size.
    
    Parameters:
    - num_steps_tunning : Number of different hyperparameter configurations to test
    - num_examples: Total number of examples used (training or evaluation)
    - num_epochs: Number of times the dataset is passed through the model
    - batch_size: Number of samples per training step
    - D: Model dimension (dimension of the tokens)
    - L: Vocabulary size (output dimension)
    - N: Sequence length (tokens per input)
    - h_u: Hidden layer size in MLP
    - H: Number of attention heads
    - K: Number of transformer layers
    - training: Whether in training mode (True) or inference (False)
    
    Returns:
    - Total FLOPs for the training/inference process
    """

    if training:
        # Training: Forward + Backpropagation
        return num_steps * flops_forward_and_back(batch_size, D, L, N, h_u, H, K)
    else:
        # Inference: Only Forward Pass
        return num_steps * flops_forward_pass(batch_size, D, L, N, h_u, H, K)
    

def linear_layer(data_dim:tuple, weight_dim:tuple, biases_dim = (0,0), bias=False):
    #X(D,N), W(h,D), B(N,h) example numbers

    #Multiplication of two matrices WX
    flops = data_dim[1]*weight_dim[0]*(2*data_dim[0]-1)

    if  bias:
        # addition of WX+B (N,h)dim
        flops += biases_dim[1]*biases_dim[0] 
        
    return flops

def RMSNorm_layer(data_dim:tuple):
    #Assume X(D,N)
    D,N=data_dim

    #N times calculation of the srt(1/d sumj(Xij**2)) where a srt is 10 flops, 1 flop addition/division/multiplication
    flops = (D+D+11)*N 

    #NxD division and a sum  (for each element in the matrix)
    flops += 2*D*N 

    #Multiplication by alpha (for each element in the matrix)
    flops += D*N 

    return flops

def softmax_flops(Q_dim:tuple):
    #We have two matrix K, Q of same dimensions 
    # Lets assume Q(d_h,N) for simplicity
    d_h, N  = Q_dim

    #Matrix multiplication S = Q^TK 
    flops = N*N*(2*d_h-1)

    #Division for each element in the matrix S(N,N) and the sqrt of d_h-->T(N,N)
    flops += N*N + 10

    # For each element in T we do exp(Tij)/sumj(exp(Tij))
    flops += N*(N*11) #For each column we compute the sum on the denominator, 1exp (10FLOPS) and an addition for each element in the column
    flops += (N*N)*11 #For each element we do and exp and a div

    return flops

def silu_flops(G_dim:tuple):
    #We have two matrix G, U of same dimensions 
    # Lets assume G(h_v,N) for simplicity
    h_v,N = G_dim

    #Sigmiod of G matrix 1/1+exp(-G)-->sigmiod(G)(h_v,N)
    flops = 12*N*h_v #1exp (10FLOPS) + 1add+ 1div for each element in the G matrix

    # Matrix Multiplication of P =G^Tsigmoid(G) 
    flops += N*N*(2*h_v-1)

    #Matrix mutiplication UP 
    flops += N*h_v*(2*N-1)

    return flops

def rope_layer(data_dim:tuple):
    ##ssume X(D,N)
    D,N = data_dim

    # the addition of the positional embedding layer
    flops = D*N

    return flops

def one_head_attention_flops(D, d_h, N):
    "FLOPS for one head of the Multi-Head self Attention layer"

    # We have three linear layers with biases of sames dimensions
    flops = 3*linear_layer(data_dim=(D,N), weight_dim= (d_h,D), biases_dim = (d_h,N), bias=True)

    #Softmax between Querys and keys--> S(N,N)
    flops +=  softmax_flops(Q_dim = (d_h,N))

    #Linear multiplication between VS without bias
    flops += linear_layer(data_dim=(N,N), weight_dim= (d_h,N), biases_dim = (d_h,N), bias=False)

    return flops


def multi_head_attention_flops(D, H, N):
    "FLOPS for the Multi-head Self Attention"
    d_h = D//H

    # H heads
    flops = H*one_head_attention_flops(D, d_h, N)

    #Transposing each head, concatenated and apply linear layer without bias
    flops += linear_layer(data_dim=(D,N), weight_dim= (D,D), bias=False)

    return flops


def MLP_layer(D, H, N, h_u):
    "FLOPS for one MLP layer, where each layer contains Multi-head self-attention, 2 RMSNorm and Up/Down projection with SiLU activation"
    # RMSNorm
    flops = RMSNorm_layer(data_dim=(D,N))

    #Multi-head self-attention
    flops += multi_head_attention_flops(D, H, N)

    # RMSNorm
    flops += RMSNorm_layer(data_dim=(D,N))

    # Two linear layer (Up projection) without bias--> G, U (h_u,N)
    flops += 2*linear_layer(data_dim=(D,N), weight_dim= (h_u,D), bias=False)

    #SiLU activation to G and U
    flops += silu_flops(G_dim= (h_u,N))

    # One linear layer (Down projection) without bias-->Z
    flops += linear_layer(data_dim=(h_u,N), weight_dim= (D,N), bias=False)

    return flops


def flops_forward_pass(batch_size, D, L, N, h_u, H, K):
    """Compute the FLOPs for a forward pass in the training loop.
    """
    # One foward pass K MLP layer, one RMSNorm layer and a linear layer plus bias
    return batch_size * (rope_layer(data_dim=(D,N)) + K*MLP_layer(D, H, N, h_u) + RMSNorm_layer(data_dim=(D,N)) + linear_layer(data_dim=(D,N), weight_dim=(L,D), biases_dim = (L,D), bias=True))



def flops_forward_and_back(batch_size, D, L, N, h_u, H, K):
    # (We simplify things and assume backward = 2x forward, as per instruction in coursework)
    # 1 forward pass is made in parallel for the whole batch of x samples and the average loss 
    # #of the batch is calculated, i.e. 1 backward pass is made to update the weights based on the average loss
    return flops_forward_pass(batch_size, D, L, N, h_u, H, K) + 2*flops_forward_pass(1, D, L, N, h_u, H, K)



