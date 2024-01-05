import torch



def train(model, device, train_loader, optimizer, epoch, criterion):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(input=output, target=target)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 30 == 0:
            print(f"train epoch : {epoch} [{batch_idx * len(data)} / {len(train_loader)}] ({100.0 * batch_idx / len(train_loader):.0f}) \t loss : {loss.item():.6f}")



def test(model, device, test_loader, criterion):
    
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(input=output, target=target)
            predict = output.max(1, keepdim=True)[1]
            correct += predict.eq(target.view_as(predict)).sum().item()

        test_loss /= len(test_loader.dataset)

        print(f"\ntest set : average loss : {test_loss:.4f}, accuracy : {correct} / {len(test_loader.dataset)}\t ({100.0 * correct / len(test_loader.dataset):.0f})\n")
        print("=" * 50)