def get_qwen_FLOPs (input_dimension, **kwargs):

    qwen_hyperparams = {
    'num_hidden_layers': 24, #number of transformer layers
    'num_attention_heads': 7, #heads in the multi-head SA
    'vocab_size': 151936, 
    'hidden_size': 896, #dimensionality of embeddings
    'intermediate_size': 4864 #hidden size for the SiGLU activation
}
    qwen_hyperparams.update(kwargs)

    num_layers = qwen_hyperparams['num_hidden_layers']
    H = qwen_hyperparams['num_attention_heads']
    V = qwen_hyperparams['vocab_size']
    D = qwen_hyperparams['hidden_size']
    M = qwen_hyperparams['intermediate_size']
    N = input_dimension

    total = 0
    #embedding FLOPs
    FLOPs_embedding = D * N * (2 * V - 1)

    total+= FLOPs_embedding

    #transformer block FLOPs
    FLOPs_RMSNorm = 2 * N + 22
    def get_MHSA_FLOPs (D, H, N):
        FLOPs_calc_QKV = 3 * D/H * N * (2*D-1) + D/H * N 
        FLOPs_rope = D/H*N*(2*D/H-1)    
        FLOPs_dot_prod_QK = N *N * (2*D/H-1)
        FLOPs_div_of_D_over_H = 10 * N*N
        FLOPs_softmax = 10 * N
        FLOPs_dot_prod_V_a = D/H * N * (2*N-1)
        FLOPs_output = D*N*(2*D-1)

        return (FLOPs_calc_QKV + FLOPs_rope + FLOPs_dot_prod_QK + FLOPs_div_of_D_over_H + FLOPs_softmax + FLOPs_dot_prod_V_a) * H + FLOPs_output
    
    FLOPs_MHSA = get_MHSA_FLOPs(D,H,N)
    FLOPs_res_connection = D * N
    FLOPs_SiGLU = M * N * (2*D-1)2 + 13 * M *N + M * N + D*N(M-1)

    FLOPs_transformer_block = FLOPs_RMSNorm + FLOPs_MHSA + FLOPs_res_connection + FLOPs_RMSNorm + FLOPs_SiGLU + FLOPs_res_connection 

    #MHSA FLOPs for all layers
    total += FLOPs_transformer_block * num_layers

    #norm before the output FLOPs
    total+= FLOPs_RMSNorm

    #fully connected linear layer maps from final state to logits
    FLOPs_LMHead = N * V * (2*D-1) + N * V
    total+= FLOPs_LMHead
    
    print(f'Total FLOPs: {total:.2e}')
    return total