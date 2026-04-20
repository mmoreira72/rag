---
name: antecipar-boleto
description: Use quando o participante perguntar sobre antecipar a data de pagamento de um boleto, relatar erro ao tentar antecipar boleto, ou quiser saber o prazo mínimo para antecipar um resgate do tipo pag-contas/boleto.
---
# Skill: Antecipar Boleto

## Quando usar
- Participante quer antecipar a data de vencimento de um boleto resgatado.
- Participante relata erro ao tentar mudar a data de pagamento de um boleto.
- Participante pergunta qual o prazo mínimo para antecipar.
- Participante não consegue selecionar uma data anterior no resgate.

## Regra de negócio (prazo)

> Para antecipar o pagamento de um boleto, o resgate deve ser solicitado com no mínimo **3 dias úteis de antecedência** em relação à nova data desejada, **antes das 12h** do dia da solicitação.

**Exemplo:**
- Quer pagar dia 28/02 (sexta): precisa ter feito o resgate até as 12h de terça-feira 25/02 (3 dias úteis antes).
- Se a solicitação foi feita após as 12h do dia, o prazo já não é mais suficiente para a data desejada.

## Procedimento

1. **Coletar as informações necessárias** antes de qualquer avaliação:
   - Qual a data de vencimento atual do boleto?
   - Para qual data o participante quer antecipar?
   - Quando foi feita a solicitação de resgate (data e horário)?
   - Se o participante não souber essas informações, oriente-o a verificar no extrato ou entrar em contato com o suporte operacional.

2. **Com as informações em mãos**, calcular se o prazo de 3 dias úteis ainda é viável:
   - Se sim: orientar o participante a solicitar o resgate com a nova data.
   - Se não: informar que não é mais possível para aquela data específica e explicar o motivo.

3. **Se o boleto já foi gerado com vencimento original**: infelizmente não é possível alterar.

## Mensagens de resposta

**Quando as informações ainda não estão disponíveis:**
> "Para verificar se ainda é possível antecipar, preciso saber: para qual data você quer antecipar o pagamento e quando foi feita a solicitação de resgate?"

**Quando o prazo ainda é viável:**
> "Ainda dá tempo! Para antecipar para [DATA DESEJADA], você precisa solicitar o resgate hoje até as 12h. Após esse horário, o prazo não será mais suficiente para essa data."

**Quando o prazo foi expirado:**
> "Infelizmente não dá mais para antecipar para essa data. Para antecipar, o resgate deveria ter sido solicitado antes das 12h de [DATA LIMITE], mantendo os 3 dias úteis antes do pagamento do boleto."

## Observações
- Dias úteis não incluem sábados, domingos e feriados nacionais.
- O corte das 12h é fixo — mesmo que o participante solicite às 12h01, o prazo não é mais válido para aquele dia.
- Não há exceções a essa regra — é operacional do processamento bancário.

## Exemplo real (chat Ewerthynho, 25/02/2025)
- Participante: queria antecipar boleto com vencimento 01/03 para 28/02.
- Problema: a solicitação foi feita depois das 12h do dia 25/02, impossibilitando os 3 dias úteis.
- Resolução: atendente explicou a regra e o participante compreendeu.
