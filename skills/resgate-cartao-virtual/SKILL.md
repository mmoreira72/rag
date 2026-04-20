---
name: resgate-cartao-virtual
description: Use quando o participante perguntar sobre resgate de vale ou cartão virtual, prazo para o saldo aparecer no aplicativo do cartão, recarga de cartão virtual, ou quando o saldo não apareceu no app após o resgate.
---
# Skill: Resgate de Cartão Virtual / Vale

## Quando usar
- Participante pergunta quando o saldo do cartão virtual vai aparecer no aplicativo.
- Participante resgatou um vale/cartão virtual e o saldo não foi liberado no app.
- Participante quer fazer uma recarga no cartão virtual que já possui.
- Participante pergunta a diferença entre primeiro resgate e recargas.

## Regra de negócio

### Primeiro resgate (cartão novo)
> O primeiro resgate de um cartão virtual exige que a equipe **solicite a liberação do saldo** junto ao parceiro. O prazo para o saldo aparecer no aplicativo é de **até 12 dias úteis** após o resgate.

### Recargas (cartão já existente)
> A partir do segundo resgate, **não é necessário solicitar nova liberação**. O saldo é liberado **diretamente no aplicativo** pelo parceiro, sem intervenção do atendimento.

## Procedimento

1. **Confirmar o ICARE** do participante.
2. **Identificar se é primeiro resgate ou recarga**:
   - Primeiro resgate: verificar se a solicitação de liberação já foi feita. Se não, acionar o parceiro.
   - Recarga: orientar que o saldo será liberado diretamente no app, sem necessidade de chamado.
3. **Informar o prazo** conforme o tipo.
4. **Se o prazo já passou e o saldo não apareceu**: escalar para o parceiro.

## Mensagens padrão

Primeiro resgate (liberação em andamento):
> "Como é o seu primeiro resgate desse cartão, precisamos solicitar a liberação do saldo junto ao parceiro. O prazo é de até 12 dias úteis para aparecer no aplicativo."

Recarga (saldo automático):
> "Para recargas, o saldo é liberado diretamente no aplicativo pelo parceiro — não precisa solicitar nada. Aguarde o prazo do parceiro para ver o saldo disponível no app."

## Observações
- O cartão virtual em si (o número/dados do cartão) já fica disponível no portal assim que o resgate é feito.
- O saldo é que tem o prazo de liberação no aplicativo do cartão.
- 12 dias úteis = aproximadamente 17 dias corridos (desconsiderando feriados).

## Exemplo real (chat Stefania, 25/02/2025)
- Participante: resgates de cartão virtual, questionou sobre liberação de saldo.
- Atendente: esclareceu que primeiro resgate precisa de solicitação (12 dias úteis) e recargas vão direto no app.
