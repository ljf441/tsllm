class Flopper:
    def __init__(self, sequence, model, num_steps_training=None, batch_size=None):
        self.sequence_length = len(sequence)
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
        self.total_flops = 0

    def __str__(self):
        return f"{self.total_flops:.4e}"
    
    def matrix_multiplication(self, m, n, p, bias=False):
        flops = m*p*(2*n-1)
        if bias:
            flops += m*p
        return flops
    
    def embedding(self):
        flops = self.matrix_multiplication(self.hidden_size, self.vocab_size, self.sequence_length, bias=False)
        return flops
    
    def rms_layernorm(self, m, n):
        #self.hidden_size, self.sequence_length
        # flops = ((m*1 + 
        #           m-1 + #addition
        #           1 + #
        #           10 + #sqrt
        #           m*2 * 2
        #           )*2*n # twice for each token
        #           )
        
        # Compute the RMS:
        flops = (m*n +      #square each element
                 m*n-1 +    #summing of elements
                 m*n +      #divide elements
                 m*10       #sqrt
        )
        
        #Layer norm:
        flops += (m*n +     #division by RMS
                  m*n +     #scaling multiplication
                  m*n       #shifting addition
        )


        # flops = (m*n + #square
        #          m*n-1 + #addition
        #          m*10 + #sqrt
        #          m*n #division
        # )

        # flops = (m+m+11)*n
        # flops += 2*m*n
        # flops += m*n
        return flops
    
    def softmax(self, m, n):
        #start with two matrices. multiply them together.
        flops = self.matrix_multiplication(n, m, n) #n*n*(2*m-1) nxn matrix
        # seq(seq*10 + seq*1 + seq-1)
        # reuse exponentials; 10 flops for exp, 1 flop for addition, 1 flop for division
        # 1 addition & division for each element
        # add n-1 times for each column
        flops += n*n*10         #exp for each element
        flops += n*(n-1)        #sum elements addition
        flops += n*n            #division

        # flops += n*n + 10 #division + sqrt
        # flops += n*(n-1) #addition
        # flops += n*n*10 #exp for each elementt
        return flops
    
    def silu(self, m, n):
        #self.ffw_size, self.sequence_length
        #calculating sigmoid
        flops = 13*n*m #exp + addition + multiplication + division
        #elementwise multiplication
        # flops += n*m*1
        flops += n*n*(2*m-1)
        #matrix multiplication up
        flops += n*m*(2*n-1)
        return flops

    def rope(self, m, n):
        #addition
        flops = m*n
        return flops
    
    def multi_head_attention(self):
        # qkv multiplications
        flops = 3 * self.matrix_multiplication(self.sequence_length, self.hidden_size, self.head_dim, bias=True)
        # softmax
        flops += self.softmax(self.head_dim, self.hidden_size)
        # weighted values
        flops += self.matrix_multiplication(self.sequence_length, self.sequence_length, self.head_dim, bias=False)
        # per head
        flops *= self.num_heads
        # concatenation, applying linear layer
        flops += self.matrix_multiplication(self.sequence_length, self.hidden_size, self.hidden_size, bias=False)
        return flops

    def ffd(self):
        # rms layer norm
        flops = self.rms_layernorm(self.hidden_size, self.sequence_length)
        # multi head attention
        flops += self.multi_head_attention()
        # rms layer norm
        flops += self.rms_layernorm(self.hidden_size, self.sequence_length)
        # gate projection and up projection have same dimensions, so combining FLOP calculations
        flops += 2 * self.matrix_multiplication(self.sequence_length, self.hidden_size, self.ffw_size, bias=False)
        # silu
        flops += self.silu(self.ffw_size, self.sequence_length)
        # down projection
        flops += self.matrix_multiplication(self.sequence_length, self.ffw_size, self.hidden_size, bias=False)
        # residuals
        flops += 2 * self.sequence_length * self.hidden_size * 1
        return flops
    
    def forward_pass(self, new_batch_size=None):
        #forward pass
        if new_batch_size is not None:
            batch_size = new_batch_size
        else:
            batch_size = self.batch_size
        flops = self.rope(self.hidden_size, self.sequence_length)
        flops += self.num_layers*self.ffd()
        flops += self.rms_layernorm(self.hidden_size, self.sequence_length)
        flops += self.matrix_multiplication(self.sequence_length, self.hidden_size, self.vocab_size, bias=True)
        flops *= batch_size
        return flops

    def forward_and_back(self):
        return self.forward_pass() + 2*self.forward_pass(new_batch_size=1)
    
    def compute_flops(self):
        self.total_flops = self.num_steps_training*self.forward_and_back()
        return self.total_flops
    
model_flopper = Flopper(np.zeros(768), model, num_steps_training=200).compute_flops()
model_flopper.compute_flops()
print(model_flopper*)