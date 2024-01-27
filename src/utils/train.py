def train_model(model, train_loader, loss_function, optimizer, n_epochs):
    """Train the model and return epoch and loss data."""
    small_difference_count, avg_loss = 0, 0
    epochs, loss_values = [], []

    for epoch in range(n_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = loss_function(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        current_avg_loss = total_loss / len(train_loader)
        # if abs(avg_loss - current_avg_loss) < 0.0001:
        #     small_difference_count += 1
        # else:
        #     small_difference_count = 0  # Reset counter if there's a significant change
        #
        # # Early stopping check
        # if small_difference_count >= 10:
        #     print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")
        #     print(f"Stopped training in epoch {epoch + 1} due to too little loss changes")
        #     break

        avg_loss = current_avg_loss

        epochs.append(epoch + 1)
        loss_values.append(avg_loss)

    return epochs, avg_loss
