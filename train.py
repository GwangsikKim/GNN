#%%
"""
def train(model, Loss, optimizer, num_epochs):
  train_loss_arr = []
  test_loss_arr = []

  best_test_loss = 99999999
  early_stop, early_stop_max = 0., 10.

  for epoch in range(num_epochs):

    # Forward Pass
    model.train()
    output = model(features)
    train_loss = criterion(output[idx_train], labels[idx_train])

    # Backward and optimize
    train_loss.backward()
    optimizer.step()
        
    train_loss_arr.append(train_loss.data)
    
    if epoch % 10 == 0:
        model.eval()
        
        output = model(features)
        val_loss = criterion(output[idx_val], labels[idx_val])
        test_loss = criterion(output[idx_test], labels[idx_test])
        
        val_acc = accuracy(output[idx_val], labels[idx_val])
        test_acc = accuracy(output[idx_test], labels[idx_test])
        
        test_loss_arr.append(test_loss)
        
        if best_ACC < val_acc:
            best_ACC = val_acc
            early_stop = 0
            final_ACC = test_acc
            print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}, Test ACC: {:.4f} *'.format(epoch, 100, train_loss.data, test_loss, test_acc))
        else:
            early_stop += 1

            print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}, Test ACC: {:.4f}'.format(epoch, 100, train_loss.data, test_loss, test_acc))

    if early_stop >= early_stop_max:
        break
        
  print("Final Accuracy::", final_ACC)

#%%
