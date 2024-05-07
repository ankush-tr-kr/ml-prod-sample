from model.preprocess import get_data_loaders


def test_data_loader_lengths():
    """Test to check the data loaders' lengths."""
    train_loader, val_loader = get_data_loaders(batch_size=64)
    assert len(train_loader.dataset) == 60000
    assert len(val_loader.dataset) == 10000
