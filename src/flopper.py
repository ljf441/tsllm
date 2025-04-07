import argparse

class Flopper:
    def __init__(self, sequence_length, model=None, num_steps_training=None, batch_size=None, use_lora=False, lora_rank=None):
        """
        Initialise Flopper object with model parameters.
        
        Args:
            sequence_length (int): Length of the input sequence.
            model (str): Model name or path.
            num_steps_training (int): Number of training steps.
            batch_size (int): Batch size.
            use_lora (bool): Use LoRA.
            lora_rank (int): LoRA rank.

        Returns:
            None
        """
        
        self.sequence_length = sequence_length

        if model ==  None:
            self.hidden_size = 896
            self.num_layers = 24
            self.num_heads = 14
            self.vocab_size = 151936
            self.ffw_size = 4864
        else:
            self.model = model
            self.model_dict = self.model.config.to_dict()
            self.hidden_size = self.model_dict['hidden_size']
            self.num_layers = self.model_dict['num_hidden_layers']
            self.vocab_size = self.model_dict['vocab_size']
            self.ffw_size = self.model_dict['intermediate_size']
            self.num_heads = self.model_dict['num_attention_heads']

        self.head_dim = self.hidden_size // self.num_heads

        if num_steps_training is None:
            num_steps_training = 10000
        if batch_size is None:
            batch_size = 4
        self.num_steps_training = num_steps_training
        self.batch_size = batch_size

        self.use_lora = use_lora
        if lora_rank == None:
            self.lora_rank = 8
        self.lora_rank = lora_rank

        self.total_flops = 0
        

    def __str__(self):
        """
        Format the FLOPs output for printing.
        """
        return f"{self.total_flops:.3e}"
    
    def matrix_multiplication(self, m, n, p, bias=False):
        """
        Calculate FLOPs for matrix multiplication.
        
        Args:
            m (int): Number of rows in the first matrix.
            n (int): Number of columns in the first matrix and rows in the second matrix.
            p (int): Number of columns in the second matrix.
            bias (bool): Whether to include bias in the calculation.
        Returns:
            flops (int): FLOPs for matrix multiplication.
        """
        flops = m*p*(2*n-1)
        if bias:
            flops += m*p
        return flops
    
    def embedding(self):
        """
        Calculate FLOPS for embedding layer.
        
        Args:
            None
        Returns:
            flops (int): FLOPs for embedding layer.
        """
        flops = self.matrix_multiplication(self.hidden_size, self.vocab_size, self.sequence_length, bias=False)
        return flops
    
    def rms_layernorm(self, m, n):
        """
        Calculate FLOPS for RMS layer normalization.
        
        Args:
            m (int): Number of rows (hidden size).
            n (int): Number of columns (sequence length).
        Returns:
            flops (int): FLOPs for RMS layer normalization.
        """
        #self.hidden_size, self.sequence_length
            
        # Compute the RMS:
        flops = (m*n +      #square each element
                 m*(n-1) +  #summing of squared elements
                 m +        #multiply by 1/hidden_size
                 m*10       #sqrt
        )
        
        #Layer norm:
        flops += (m*n +     #division by RMS
                  m*n +     #scaling multiplication factor
                  m*n       #shifting addition of epsilon
        )

        return flops
    
    def softmax(self, m):
        """
        Calculate FLOPS for softmax function.
        
        Args:
            m (int): Number of rows and columns (sequence length).
        Returns:
            flops (int): FLOPs for softmax function.
        """
        #self.sequence_length
        # seq(seq*10 + seq*1 + seq-1)

        # reuse exponentials; 10 flops for exp, 1 flop for addition, 1 flop for division
        # 1 addition & division for each element
        # add n-1 times for each column
        flops = m*m*10      #exp for each element
        flops += m*(m-1)    #sum elements addition
        flops += m*m        #division
        return flops
    
    def silu(self, m, n):
        """
        Calculate FLOPS for SiLU activation function.
        
        Args:
            m (int): Number of rows (intermediate size).
            n (int): Number of columns (sequence length).
        Returns:
            flops (int): FLOPs for SiLU activation function.
        """
        #self.ffw_size, self.sequence_length
        #calculating sigmoid
        flops = 13*n*m      #exp + addition + multiplication + division
        return flops

    def rope(self, m, n):
        """
        Calculate FLOPS for rotary positional embedding.
        
        Args:
            m (int): Number of rows (hidden size).
            n (int): Number of columns (sequence length).
        Returns:
            flops (int): FLOPs for rotary positional embedding.
        """
        #self.hidden_size, self.sequence_length
        #cost of adding rotary positional embeddings
        flops = m*n
        return flops

    def lora(self, m, r):
        """
        Calculate FLOPS for adding LoRA matrices.
        
        Args:
            m (int): Number of rows (hidden size).
            r (int): LoRA rank.
        Returns:
            flops (int): FLOPs for LoRA matrices.
        """
        #self.head_dim, self.lora_rank
        flops = m*m*2*r
        return flops
    
    def multi_head_attention(self):
        """
        Calculate the FLOPS for multi-head attention.

        Args:
            None
        Returns:
            flops (int): FLOPs for multi-head attention.
        """
        # qkv multiplications
        if self.use_lora == False:
            flops = 3 * self.matrix_multiplication(self.sequence_length, self.hidden_size, self.head_dim, bias=True)
        else:
            flops = 2 * self.lora(self.head_dim, self.lora_rank) #for LoRA linear layers
            flops += 3*self.matrix_multiplication(self.sequence_length, self.hidden_size, self.head_dim, bias=True) # q, k, v
        # attention score (matrix multiplication)
        flops += self.matrix_multiplication(self.sequence_length, self.head_dim, self.sequence_length, bias=False)
        # scaled self-attention (division and sqrt)
        flops += self.sequence_length*self.sequence_length * 1 + 10
        # softmax
        flops += self.softmax(self.sequence_length)
        # linear multiplication of matrices
        flops += self.matrix_multiplication(self.sequence_length, self.head_dim, self.sequence_length, bias=False)
        # per head
        flops *= self.num_heads
        # concatenation has no associated FLOPs, calculating output
        flops += self.matrix_multiplication(self.sequence_length, self.head_dim, self.sequence_length, bias=False)
        return flops

    def ffd(self):
        """
        Calculate FLOPS for feed-forward multi-layer perceptron.

        Args:
            None
        Returns:
            flops (int): FLOPS for the feed-forward MLP.
        """
        # rms layer norm
        flops = self.rms_layernorm(self.hidden_size, self.sequence_length)
        # multi head attention
        flops += self.multi_head_attention()
        # rms layer norm
        flops += self.rms_layernorm(self.hidden_size, self.sequence_length)
        # gate projection
        flops += self.matrix_multiplication(self.sequence_length, self.hidden_size, self.ffw_size, bias=False)
        # up projection
        flops += self.matrix_multiplication(self.sequence_length, self.hidden_size, self.ffw_size, bias=False)
        # SiLU
        flops += self.silu(self.ffw_size, self.sequence_length)
        # element-wise multiplication with SiLU activations
        flops += self.sequence_length * self.ffw_size * 1
        # down projection
        flops += self.matrix_multiplication(self.sequence_length, self.ffw_size, self.hidden_size, bias=False)
        # residuals (addition)
        flops += 2 * self.sequence_length * self.hidden_size * 1
        return flops
    
    def forward_pass(self, new_batch_size=None):
        """
        Calculates FLOPS for single forward pass. Can also be adapted for calculating the backward pass.

        Args:
            new_batch_size (int): batch size when computing FLOPS for backpropagation.

        Returns:
            flops (int): FLOPS for single pass.
        """
        #forward pass
        if new_batch_size is not None:
            batch_size = new_batch_size
        else:
            batch_size = self.batch_size
        # cost of adding the rotational positional embedding
        flops = self.rope(self.hidden_size, self.sequence_length)
        # MLP per layer
        flops += self.num_layers*self.ffd()
        # final rms layernorm
        flops += self.rms_layernorm(self.hidden_size, self.sequence_length)
        # output layer
        flops += self.matrix_multiplication(self.sequence_length, self.hidden_size, self.vocab_size, bias=True)
        # per batch
        flops *= batch_size
        return flops

    def forward_and_back(self):
        """
        Calculates FLOPS for forward and backward pass.

        Args:
            None
        Returns:
            flops (int): total FLOPS for forward and backward pass.
        """
        # backpropagation is assumed to be twice the forward pass
        return self.forward_pass() + 2*self.forward_pass(new_batch_size=1)
    
    def compute_flops(self):
        """
        Calculates total FLOPS when training.

        Args:
            None
        Returns:
            total_flops (int): total FLOPS when training.
        """
        # forward and backpropagation is computed for every training step
        self.total_flops = self.num_steps_training*self.forward_and_back()
        return self.total_flops

    def compute_validation(self):
        """
        Calculates total FLOPS when model is calculating validation loss.

        Args:
            None
        Returns:
            total_flops (int): total FLOPS when calculating validation loss.
        """
        # when computing validation loss, model only goes through the forward pass
        self.total_flops = self.forward_pass()
        return self.total_flops
    
    def compute_inference(self):
        """
        Calculates total FLOPS when model is in inference mode to predict one sequence.

        Args:
            None
        Returns:
            total_flops (int): total FLOPS when in inference mode.
        """
        # when in inference mode, the model is only passed one sequence at a times
        self.sequence_length = 1200 # one full sequence has a maximum of around 1200 tokens
        self.total_flops = self.forward_pass(new_batch_size=1)
        return self.total_flops
    

def parse_args():
    parser = argparse.ArgumentParser(description="Compute FLOPs for a specified Qwen model")
    parser.add_argument("-s", "--sequence_length", type=int, default=512, help="Sequence length of the model")
    parser.add_argument("-n", "--num_steps_training", type=int, default=10000, help="Number of training steps")
    parser.add_argument("-b", "--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("-l", "--use_lora", type=bool, default=False, help="Use LoRA")
    parser.add_argument("-r", "--lora_rank", type=int, default=None, help="LoRA rank")
    parser.add_argument("-m", "--mode", type=str, default="T", help="Training [T]; Validation [V]; Inference [I]")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_flops = Flopper(sequence_length=args.sequence_length, model=None, num_steps_training=args.num_steps_training, batch_size=args.batch_size, use_lora=args.use_lora, lora_rank=args.lora_rank)
    mode = args.mode
    if (mode.capitalize() == "T"):
        model_flops.compute_flops()
        print(model_flops)
    elif (mode.capitalize() == "V"):
        total = model_flops.compute_validation() * args.num_steps_training//50
        print(f"{total:.4e}")
    elif (mode.capitalize() == "I"):
        model_flops.compute_validation()
        print(model_flops)
    else:
        print("ERROR: INCORRECT MODE \nCHOOSE BETWEEN [T]RAINING; [V]ALIDATION; [I]NFERENCE")