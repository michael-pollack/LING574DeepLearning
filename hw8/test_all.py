import pytest
import torch

from model import Seq2SeqModel


class TestSeq2Seq:

    embedding_dim = 2
    hidden_dim = 4
    num_layers = 2
    vocab_size = 3
    padding_index = 0

    source = torch.LongTensor([[1, 1, 0, 0], [2, 1, 2, 1]])
    target = torch.LongTensor([[2, 1, 0], [1, 2, 1]])
    lengths = ([2, 4], [2, 3])

    @pytest.fixture()
    def model(self):
        the_model = Seq2SeqModel(
            TestSeq2Seq.embedding_dim,
            TestSeq2Seq.hidden_dim,
            TestSeq2Seq.num_layers,
            TestSeq2Seq.vocab_size,
            TestSeq2Seq.padding_index,
        )
        the_model.load_state_dict(torch.load("test_model.pt"))
        return the_model

    def test_forward(self, model):
        forwarded = model.forward(
            TestSeq2Seq.source, TestSeq2Seq.target, TestSeq2Seq.lengths
        )
        forwarded_gold = torch.load("test_forward.pt")
        assert torch.allclose(forwarded, forwarded_gold)

    def test_encode(self, model):
        embeddings = model.embedding(TestSeq2Seq.source)
        lengths = TestSeq2Seq.lengths[0]
        encoded = model.encode(embeddings, lengths)
        encoded_gold = torch.load("test_encode.pt")
        # check output equal
        assert torch.allclose(encoded[0], encoded_gold[0])
        # check h equal
        assert torch.allclose(encoded[1][0], encoded_gold[1][0])
        # check c equal
        assert torch.allclose(encoded[1][1], encoded_gold[1][1])

    def test_decode(self, model):
        target_embeddings = model.embedding(TestSeq2Seq.target)
        target_lengths = TestSeq2Seq.lengths[1]
        encoded_gold = torch.load("test_encode.pt")
        encoder_states = encoded_gold[0]
        inits = encoded_gold[1]
        attention_mask = model.get_padding_mask(
            batch_size=TestSeq2Seq.source.size(0),
            max_key_len=TestSeq2Seq.source.size(1),
            key_lengths=TestSeq2Seq.lengths[0],
        ).float()
        decoded = model.decode(
            inits, target_embeddings, target_lengths, encoder_states, attention_mask
        )
        decoded_gold = torch.load("test_decode.pt")
        # check output equal
        assert torch.allclose(decoded[0], decoded_gold[0])
        # check h equal
        assert torch.allclose(decoded[1][0], decoded_gold[1][0])
        # check c equal
        assert torch.allclose(decoded[1][1], decoded_gold[1][1])

    def test_attention(self, model):
        decoded = torch.load("test_decoder_out.pt")
        encoder_states = torch.load("test_encode.pt")[0]
        attention_mask = model.get_padding_mask(
            batch_size=TestSeq2Seq.source.size(0),
            max_key_len=TestSeq2Seq.source.size(1),
            key_lengths=TestSeq2Seq.lengths[0],
        ).float()
        attention = model.attention(decoded, encoder_states, padding_mask=attention_mask)
        attention_gold = torch.load("test_attention.pt")
        assert torch.allclose(attention, attention_gold)
