def train_model(model, train_loader, optimizer, criterion, device, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
