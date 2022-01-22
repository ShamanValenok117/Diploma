#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# torch
import torch
from tqdm import tqdm


# In[ ]:


SEP_LEVELS = int(2.65 * 1000 )

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.linear = torch.nn.Linear(300,300, bias = True, dtype=torch.float32)
        
        self.emb_lvl = torch.nn.Embedding(SEP_LEVELS,300)
        #self.linear_attention = torch.nn.Linear(300,300, dtype=torch.float32)
        
        self.lstm = torch.nn.LSTM(input_size=300, hidden_size=128,batch_first = True, bidirectional=True) #dropout=.3,num_layers=2
        self.linear_1 = torch.nn.Linear(in_features=128, out_features = 1, bias = True)
        
    def forward(self,x,previous_y):
                        
        # pass pipe pressures lags through Dense 300 ->300
        after_linear = torch.nn.functional.relu( self.linear(x) )
        
        # get previous separator level embedding
        previous_separator_level = round(previous_y ,3) * 1000      # previous_y = y[0]
        previous_separator_level = int( previous_separator_level ) 
        previous_separator_level = torch.tensor( [ previous_separator_level ] )
        previous_separator_level_emb = self.emb_lvl(previous_separator_level)     # Делаем ему размер emb = 300
        previous_separator_level_emb = previous_separator_level_emb.reshape(1,300)
        
        # get attention
        attention = after_linear * previous_separator_level_emb
        #attention = self.linear_attention(attention)
        attention_softmax = torch.nn.functional.softmax( attention , dim=0 )
        
        # multiply presure lags with attention softmax 
        final_state = x * attention_softmax
        final_state = final_state.reshape(1,5,300)
        
        # LSTM
        embeddings, (shortterm, longterm) = self.lstm(final_state)
        longterm = torch.add(longterm[0],longterm[1])/2
        predict = self.linear_1(longterm)
        
        return predict
        
    def fit(self, X_features, y_target):
    
        EPOCHS = 50 
        criteria = torch.nn.HuberLoss()
        ONE_EPOCH_SIZE = len(y_target) - 5
        min_loss = 0.2
        true_array = torch.from_numpy( y_target[5:]).float()
        lr= 0.001

        for epoch in range(EPOCHS):
            prediction_array = torch.tensor( [] ).float()

            for i in tqdm(  range(5, len(y),1 )  ):
                y_pred_array = torch.tensor([])
                for j in range(3):
                    x = np.array(X_features.iloc[i-5:i,j*300:(j+1)*300]) 
                    x = torch.from_numpy(x).reshape(5,300).float()

                    previous_y = y_target[i]

                    y_pred = self.forward( x,previous_y )
                    y_pred_array = torch.cat((y_pred_array,y_pred) )
                y_pred = torch.mean(y_pred_array).reshape(-1)

                prediction_array = torch.cat((prediction_array,y_pred), dim=0)

            loss = criteria( prediction_array.reshape(-1), true_array)

            if loss < min_loss:
                min_loss = loss
                torch.save(net,f'./model_loss_{loss}')
                lr = 0.0001

            loss.backward()

            optimizer = torch.optim.Adam( net.parameters(), lr = lr)


            optimizer.step() # обновляем веса модели
            optimizer.zero_grad()  # обнуляем веса

            # at the end of each epoch print the last loss
            if epoch % 1 ==0: print(f'epoch {epoch}. MSELoss = {loss}')
    
    @torch.no_grad()
    def predict(self,X_features):
        prediction_array = np.array([0.],dtype=np.float32)

        #with torch.no_grad():
        for i in tqdm( range(5, len(X_features), 1 ) ): 
            y_pred_array = torch.tensor([])
            for j in range(3):
                x = np.array(X_features.iloc[i-5:i , j*300:(j+1)*300]) 
                x = torch.from_numpy(x).reshape(5,300).float()

                previous_y = prediction_array[-1]
                
                y_pred = self.forward( x,previous_y )
                y_pred_array = torch.cat((y_pred_array,y_pred) )
            y_pred = abs(torch.mean(y_pred_array))

            prediction_array = np.append( prediction_array, y_pred.numpy() )
        return prediction_array[1:]
     
        
        
        
        
net = Net()
print(net)

