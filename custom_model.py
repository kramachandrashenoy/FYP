import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from transformers import BertTokenizerFast

# Custom Transformer Encoder-Decoder Model
class CustomTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, ff_size, num_layers, max_length):
        super(CustomTransformer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_size, num_heads, ff_size, batch_first=True),
            num_layers
        )
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_size, num_heads, ff_size, batch_first=True),
            num_layers
        )
        
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Adjust sequence lengths dynamically
        src = src[:, :MAX_LENGTH]
        tgt = tgt[:, :MAX_LENGTH]
        
        # Positional encoding
        src_positions = torch.arange(0, src.size(1)).unsqueeze(0).to(src.device)
        tgt_positions = torch.arange(0, tgt.size(1)).unsqueeze(0).to(tgt.device)

        src_embedding = self.token_embedding(src) + self.position_embedding(src_positions)
        tgt_embedding = self.token_embedding(tgt) + self.position_embedding(tgt_positions)

        src_embedding = self.dropout(src_embedding)
        tgt_embedding = self.dropout(tgt_embedding)

        memory = self.encoder(src_embedding, src_key_padding_mask=src_mask)
        output = self.decoder(tgt_embedding, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)

        return self.fc_out(output)

# Hyperparameters
VOCAB_SIZE = 30522  # Vocabulary size (e.g., BERT tokenizer vocab size)
EMBED_SIZE = 512
NUM_HEADS = 8
FF_SIZE = 2048
NUM_LAYERS = 6
MAX_LENGTH = 256

# Initialize the model
custom_model = CustomTransformer(VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, FF_SIZE, NUM_LAYERS, MAX_LENGTH)

# Dataset Class
class RealSummarizationDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_length=256):
        self.tokenizer = tokenizer
        self.inputs, self.targets = self._load_dataset(data_path)
        self.max_length = max_length

    def _load_dataset(self, path):
        inputs = []
        targets = []
        judgement_path = os.path.join(path, "train-data", "judgement")
        summary_path = os.path.join(path, "train-data", "summary")

        for fname in os.listdir(judgement_path):
            with open(os.path.join(judgement_path, fname), 'r', encoding='utf-8') as f:
                inputs.append(f.read())
            with open(os.path.join(summary_path, fname), 'r', encoding='utf-8') as f:
                targets.append(f.read())

        return inputs, targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_enc = self.tokenizer(self.inputs[idx], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        target_enc = self.tokenizer(self.targets[idx], max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        return input_enc['input_ids'].squeeze(), target_enc['input_ids'].squeeze()

# Dataset Path
DATASET_PATH = r"C:\\Users\\Ramachandra\\OneDrive\\Desktop\\FYP\\dataset (3)\\dataset\\IN-Abs"

# Load tokenizer and dataset
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
real_dataset = RealSummarizationDataset(tokenizer, DATASET_PATH)

# Create DataLoader
real_dataloader = DataLoader(real_dataset, batch_size=2, shuffle=True)

# Training
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.AdamW(custom_model.parameters(), lr=5e-5)

epochs = 3
for epoch in range(epochs):
    for src, tgt in real_dataloader:
        src, tgt = src.to("cpu"), tgt.to("cpu")
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        outputs = custom_model(src, tgt_input)
        outputs = outputs[:, :tgt_output.size(1), :]

        # Ensure outputs are contiguous before reshaping
        loss = criterion(outputs.contiguous().view(-1, VOCAB_SIZE), tgt_output.contiguous().view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Example Inference
custom_model.eval()
example_text = '''N: Criminal Appeal No. 8 of 1951.
Appeal from an Order of the High Court of Bombay (Bavdekar and Chainani JJ.) dated 20th February, 1950, in Criminal Appeal No. 106 of 1950 arising out of an order dated 9th January, 1950, of the Presidency Magistrate, 19th Court, Esplanade, Bombay, in Case No. 10879/P of 1949.
The facts are stated in the judgment.
Iswarlal C. Dalai and R.B. Dalai, for the appellant.
C.K. Daphtary, Solicitor General for India (G. N. Joshi, with him) for the Republic of India (respondent No. 1).Jindra Lal for the respondent No. 2. 1952.
February 1.
The Judgment of the Court was deliv ered by CHANDRASEKHARA AIYAR J.
The facts out of which this Crimi nal Appeal has arisen are not long.
The appellant, W.H. King, who is carrying on a business in Bombay under the name and style of Associated Commercial Enterprises, was the tenant of a flat on the second floor of a building called "Ganga Vihar", Marine Drive, Bombay, which belongs to a lady named Durgeshwari Devi.
The tenancy was a monthly one, the rent being Rs. 215.
It is said that the appellant wanted to go to the United Kingdom for treatment of his failing eye sight and he got into touch with the complainant Mulchand Kodumal Bhatia, who is the second respondent in this appeal, through one Sayed for the purpose of making necessary ar rangements about the flat occupied by him in view of his intended departure.
The prosecution case is that the accused demanded a sum of Rs. 30,000 which was later on reduced to Rs. 29,500 as consideration for putting the complainant in vacant possession of the flat and an additional amount of Rs. 2,000 for the furniture, and that the complainant agreed to pay these sums.
The complainant actually paid the accused two sums of 420 Rs. 500 each on 7th November, 1948, and 17th November, 1948.
He, however, got into touch with the police on 1 12 1948, and in conjunction with the latter, a trap was laid for the appellant.
It was arranged that the complainant should bring with him Rs. 1,000, being the balance due in respect of the furniture and that the police would give him Rs. 29,500 to be paid to the appellant.
The complainant and a Sub Inspec tor, posing as the complainant 's brother, went to the appel lant on 4 12 1948, and paid him the two sums of money; and the keys of the flat and the motor garage were handed over to the complainant.
As the appellant and his wife were leaving the flat, the man, who masqueraded as the complain ant 's brother, threw off his disguise and disclosed his identity.
The police party, who were down below ready for the raid, held up the car of the appellant and recovered the sum of Rs. 30,500 from the rear seat of the car and also some papers, a typed draft of a partnership agreement be tween the complainant and the appellant and an application form for permission to occupy the building as caretaker.
From the complainant were recovered the bunch of keys and the documents that were handed over to him by the appellant, namely, the letter handing vacant possession (Exhibit D).
the receipt for Rs. 2,000 for the articles of furniture (Exhibit E), a letter to the Bombay Gas Company for transfer of the gas connection to the name of the complainant (Exhib it F), and the letter to the Bombay Electric Supply and Transport Committee for transfer of the telephone connec tions and the deposit of Rs. 27 (Exhibit G).
The appellant was charged under section 18(1) of the Bombay Rents, Hotel and Lodging House Rates Control Act, LVII of 1947, for receiving a pugree of Rs. 29,500 and he was further charged under section 19(2) of the said Act for receiving the said sum as a condition for the relin quishment of his tenancy.
His wife, who was the second accused in the case, was charged with aiding and abetting her husband in the commission of the two offences.
421 The defence of the appellant was that he was in search of a partner to carry on his business during his intended absence, who was also to act as caretaker of his flat anal that it was in this connection and with this object in view that he entered into negotiations with the complain ant.
The sum of Rs. 29 500 was not pugree but represented capital for 0 12 0 share in the business and as the com plainant was also to be a caretaker of the flat, the sum of Rs. 2,000 was paid and received as a guarantee against disposal and damage of the furniture and it was agreed to be paid back on the appellant 's return to India.
The wife of the appellant denied any aiding and abetting.
The Presidency Magistrate, who tried the case, disbe lieved the defence on the facts, holding that what was received by the accused was by way of pugree.
As section 18 (1) of the Act was not applicable he convicted him under section 19(2) of the Act and sentenced him, in view of his old age and blindness, to one day 's simple imprisonment and a fine of Rs. 30,000.
The wife was acquitted, the evidence being insufficient to prove any abetment.
The appellant preferred an appeal to the High Court of Bombay but it was summarily dismissed on 20 2 1950.
He asked for a certificate under article 134(1)(c) of the Constitution but this was rejected on 10 4 1950.
Thereaf ter he applied for special leave to appeal to this Court and it was granted on 3 10 1950.
A short legal argument was advanced on behalf of the appellant based on the language of section 19 (1) of the Act and this is the only point which requires our consideration.
The section which consists of two parts is in these terms:" "(1) It shall not be lawful for the tenant or any person acting or purporting to act on behalf of the tenant to claim or receive any sum or any consideration as a condi tion for the relinquishment of his tenancy of any premises; 422 (2) Any tenant or person who in contravention of the provisions of sub section (1) receives any sum or considera tion shall, on conviction, be punished with imprisonment for a term which may extend to 6 months and shall also be pun ished with fine which shall not be less than the sum or the value of the consideration received by him.
" It was urged that the offence arises only on receipt of any sum or any consideration as a condition of the relin quishment by a tenant of his tenancy and that in the present case there was no such relinquishment.
Exhibit D, which is the most material document, under which the appellant handed over vacant possession of the flat to the complainant, constitutes or evidences an assignment of the tenancy and not a relinquishment.
It says : "I, W.H. King, hereby hand over vacant possession of my flat No. 3 situated on 2nd floor and garage No. 4 on the ground floor of Ganga Vihar Building on Plot No. 55 situated on Marine Drive Road to Mr. Mulchand Kodumal Bhatia from this day onward and that I have no claim whatsoever over this flat and Mr. Mulchand Kodumal Bhatia will pay the rent directly to the landlord.
" The argument raised on behalf of the appellant appears to us to be sound and has to be accepted.
The learned Solic itor General urged that 'the word "relinquishment" was not a term of art and was used in the section not in any strict technical sense but in its comprehensive meaning as giving up of possession of the premises; and he pointed out that if it was intended by the legislature that "relinquish ment" should have the limited meaning sought to be placed upon it on behalf of the appellant, the word "surrender" used in the Transfer of Property Act would have been more appropriate.
Sections 15 and 18 of the Act were referred to in this connection but in our opinion they lend no assist ance to the argument of the learned counsel.
Any sublet ting, assignment or transfer in any other manner of his interest by the tenant is made unlawful under 423 section 15.
Section 18 deals with the grant, renewal or continuance of a lease of any premises or the giving of his consent by the landlord to the transfer of a lease by sub lease or otherwise, and it provides that the landlord, who receives any fine, premium, or other like sum or deposit, or any consideration for the grant, renewal or continuance or the accord of consent oh would be guilty of an offence and liable to the punishment therein specified.
It would thus be seen that an assignment of the lease or transfer in any other manner by a tenant is not made an offence; the statute merely says that it is not a lawful transaction.
It is the landlord 's consent to the transfer of a lease by sub lease or otherwise on receipt of consideration that has been made an offence.
Then follows section 19 which speaks of the relinquishment of his tenancy of any premises by a tenant.
If, by the expression, an assignment such as we have in the present case was meant, appropriate words could have been used, such as the transfer by a tenant of his interest, which we find in section 108, sub clause (i), of the Trans fer of Property Act.
The distinction between an assignment on the one hand and relinquishment or surrender on the other is too plain to be ignored.
In the case of an assignment, the assignor contin ues to be liable to the landlord for the performance of his obligations under the tenancy and this liability is contrac tual, while the assignee becomes liable by reason of privity of estate.
The consent of the landlord to an assignment is not necessary, in the absence of a contract or local usage to the contrary.
But in the case of relinquishment, it cannot be a unilateral transaction; it can only be in favour of the lessor by mutual agreement between them.
The relin quishment of possession must be to the lessor or one who holds his interest.
In fact, a surrender or relinquishment terminates the lessee 's rights and lets in the lessor.
It is no doubt true that the word "relinquishment" does not occur in the Transfer of Property Act but it is found in many of the Tenancy Acts in various provinces where there are Sec tions which deal with the 55 424 relinquishment of their holdings by tenants in favour of the landlord by notice given to him in writing.
The section in question, it should be further noted, does not speak of relinquishment or giving up of possession,in general terms.
The words are "the relinquishment of his tenancy of any premises".
The relinquishment of a tenancy is equivalent to surrender by the lessee or tenant of his rights as such.
Whether abandonment of a tenancy would come within the meaning of relinquishment is a question that does not arise in this appeal, because in the face of Exhibit D, there is no abandonment in the sense that the tenant disappeared from the scene altogether saying nothing and making no arrange ments about his interest and possession under the lease.
As the statute creates an offence and imposes a penalty of fine and imprisonment, the words of the section must be strictly construed in favour of the subject.
We are not concerned so much with what might possibly have been intend ed as with what has been actually said in and by the language employed.
As in our view, there has been no "relinquishment" within the meaning of section 19, sub clause (1), the conviction under sub clause (2) cannot be sustained.
It is set aside and the fine of Rs. 30,000 will be refunded if it has al ready been paid.
The other parts of the order of the learned Presidency Magistrate, as regards the disposal of Rs. 1,000 paid by the complainant to the appellant and the sum of Rs. 29,500 brought in by the police, will, however, stand.
Conviction sit aside.
Agent for respondent No. 1: P.A. Mehta.
Agent for respondent No. 2: Ganpat Rai.'''
tokenized_input = tokenizer(example_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True, padding="max_length")
input_ids = tokenized_input["input_ids"]

# Generate summary
with torch.no_grad():
    memory = custom_model.encoder(custom_model.token_embedding(input_ids))
    output_ids = torch.zeros((1, MAX_LENGTH), dtype=torch.long)
    for i in range(1, MAX_LENGTH):
        decoder_output = custom_model.decoder(
            custom_model.token_embedding(output_ids[:, :i]), memory
        )
        logits = custom_model.fc_out(decoder_output)
        next_token = logits[:, -1, :].argmax(dim=-1)
        output_ids[:, i] = next_token
        if next_token == tokenizer.sep_token_id:
            break

generated_summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated Summary:", generated_summary)
