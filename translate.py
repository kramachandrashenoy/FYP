from transformers import MBartForConditionalGeneration, MBart50Tokenizer

# Load the tokenizer and model
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50Tokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Set the source and target languages
tokenizer.src_lang = "en_XX"  # Source language: English
target_language = "ta_IN"     # Target language: Tamil

# Input text
text = '''N: Criminal Appeal No. 8 of 1951.
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

# Tokenize the input with truncation and padding
inputs = tokenizer(
    text,
    return_tensors="pt",
    max_length=256,  # Adjust the maximum length based on the model's limits
    truncation=True,
    padding=True
)

# Generate the translation
translated_tokens = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.lang_code_to_id[target_language]
)

# Decode the translated text
translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
print("Translated Text:", translated_text)


#Bengali (bn_IN), Hindi (hi_IN), Tamil (ta_IN), Telugu (te_IN), Gujarati (gu_IN), Malayalam (ml_IN), Marathi (mr_IN) etc

# Output:

'''Translated Text: N: 1951 ஆம் ஆண்டின் குற்றவியல் மேல்முறையீட்டு எண் 8. 1950 ஆம் ஆண்டு பிப்           
ரவரி 20 ஆம் தேதியிலான மும்பை உயர்நீதி மன்றத்தின் (பாவ்டெகர் மற்றும் சாய்னானி ஜே.ஜி.) குற்
றவியல் மேல்முறையீட்டு எண் 106 இல், 1950 ஆம் ஆண்டு ஜனவரி 9 ஆம் தேதியிலான மும்பை, எஸ்லனேட்,
19 ஆம் நீதிமன்றத் தலைமை நீதிபதியின், 1949 ஆம் ஆண்டு வழக்கு எண் 10879/P ல் இருந்து பெறப்ப            
பட்ட தீர்ப்பில் இருந்து மேல்முறையீட்டு. இந்திய குடியரசின் தலைமை வக்கீல் C.K. Daphtary (அவர 
ருடன் ஜி. என். ஜோஷி) இந்திய குடியரசின் தலைமை வக்கீல் C.C. Dalai மற்றும் R.B. Dalai ஆகியோரி 
ின் தீர்ப்பில் கூறப்பட்டுள்ள உண்மைகள். 1952 ஆம் ஆண்டு பிப்ரவரி 1.'''