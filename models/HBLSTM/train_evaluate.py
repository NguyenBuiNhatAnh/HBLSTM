def train_model(model, train_loader, criterion, optimizer, device):
    model.train() # Bật chế độ training
    total_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # 1. Zero gradients
        optimizer.zero_grad()
        
        # 2. Forward pass
        outputs = model(batch_X)
        
        # 3. Tính loss (outputs shape: [batch, 1], batch_y shape: [batch])
        # Cần unsqueeze batch_y thành [batch, 1] để cùng shape với outputs
        loss = criterion(outputs, batch_y.unsqueeze(1))
        
        # 4. Backward pass
        loss.backward()
        
        # 5. Cập nhật trọng số
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, criterion, device):
    model.eval() # Bật chế độ evaluation (tắt Dropout, BatchNorm nếu có)
    total_loss = 0.0
    
    # Tắt tính toán gradient để tiết kiệm bộ nhớ và chạy nhanh hơn
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            
            total_loss += loss.item()
            
    return total_loss / len(test_loader)