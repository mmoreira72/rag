---
name: resgate-cartao-virtual
description: Use quando o participante perguntar sobre resgate de vale ou cartão virtual, prazo para o saldo aparecer no aplicativo, recarga de cartão virtual ou físico, cartão chegou sem saldo, email do cartão não recebido, ou limites de recarga do cartão físico.
---
# Skill: Resgate de Cartão Virtual e Físico

## Quando usar
- Participante pergunta quando o saldo do cartão virtual vai aparecer no aplicativo.
- Participante resgatou um vale/cartão virtual e o saldo não foi liberado no app.
- Participante quer fazer uma recarga no cartão virtual que já possui.
- Participante não recebeu o email com os dados do cartão virtual.
- Participante recebeu o cartão físico mas o saldo está zerado.
- Participante quer recarregar o cartão físico.

## Regra de negócio

### Cartão virtual — Primeiro resgate (cartão novo)
> O primeiro resgate de um cartão virtual exige que a equipe **solicite a liberação do saldo** junto ao parceiro. O prazo para o saldo aparecer no aplicativo é de **até 12 dias úteis** após o resgate.

### Cartão virtual — Recargas (cartão já existente)
> A partir do segundo resgate, **não é necessário solicitar nova liberação**. O saldo é liberado **diretamente no aplicativo** pelo parceiro. Prazo: **até 12 dias úteis**.

### Email do cartão virtual não recebido
- Orientar o participante a verificar caixa de spam, lixeira, abas de promoções e capacidade da caixa.
- Se ainda não encontrado após o prazo de 12 dias úteis, solicitar reenvio.

### Cartão físico — Primeiro saldo zerado
> O primeiro saldo do cartão físico não é automático. Após o cartão existir no app, é necessário solicitar a liberação separadamente. Prazo de retorno: alguns dias úteis.

### Cartão físico — Recarga
- Existe um produto de recarga específico para o cartão físico.
- O saldo vai diretamente para o app.
- Prazo: **até 12 dias úteis**.
- Mencionar limites diário e mensal quando relevante.

## Procedimento

1. **Confirmar o ICARE** do participante.
2. **Identificar o tipo**: virtual ou físico, e se é primeiro resgate, recarga, email não recebido ou saldo zerado.
3. **Aplicar a regra** correspondente.
4. **Solicitar número do pedido** apenas quando necessário para escalar.
5. **Informar o prazo** conforme o tipo.
6. Se o prazo já passou e o saldo não apareceu: escalar para o parceiro.

## Mensagens padrão

Primeiro resgate — liberação em andamento:
> "Como é o seu primeiro resgate desse cartão, precisamos solicitar a liberação do saldo junto ao parceiro. O prazo é de até 12 dias úteis para aparecer no aplicativo."

Recarga — saldo automático:
> "Para recargas, o saldo é liberado diretamente no aplicativo pelo parceiro — não precisa solicitar nada. Aguarde até 12 dias úteis para ver o saldo disponível no app."

Cartão físico com saldo zerado:
> "O primeiro saldo do cartão físico precisa de uma liberação específica após o cartão aparecer no app. Vou verificar e acionar o parceiro se necessário."

Email não recebido:
> "Verifique a pasta de spam, lixeira e abas de promoções. Se ainda não encontrou após 12 dias úteis, me avisa que solicito o reenvio."

## Observações
- O número/dados do cartão virtual ficam disponíveis no portal assim que o resgate é feito. O saldo é que tem prazo.
- 12 dias úteis ≈ 17 dias corridos (sem feriados).
- Dias úteis não incluem sábados, domingos e feriados nacionais.

## Exemplos reais
- Stefania (25/02/2025): primeiro resgate de cartão virtual, esclarecimento sobre 12 dias úteis para saldo e diferença para recargas.
- Email não recebido: filtro por remetente + pedido de reenvio após 12 dias úteis.
- Cartão físico com saldo zerado: liberação separada acionada, retorno em dias úteis.
- Recarga de cartão físico: participante recebeu link do produto de recarga e foi informada sobre limites diário e mensal.
