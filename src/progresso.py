"""Mostra o andamento de uma fase e estima quanto falta.

Uso (na pasta dos scripts, enquanto a fase roda):
    python progresso.py --fase B --sementes 0,1,2 --paralelo 4

Lê resultados/<id>/s<semente>/*/training.json e status.json. A estimativa usa a
velocidade atual das execuções em andamento e a duração das que já terminaram,
ajustada pelo tamanho do trial (horizonte × episódios) de cada experimento.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import experimentos as exp

AVALIACAO_S = 5 * 60   # avaliação + diagnóstico depois do treino, aproximado


def ler_json(caminho, padrao=None):
    try:
        return json.loads(Path(caminho).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return padrao


def tempo_treinado(historico):
    """Soma do tempo de todas as sessões (o relógio recomeça quando a execução é retomada)."""
    total, anterior = 0.0, None
    for r in historico:
        s = r.get('seconds', 0.0)
        if anterior is not None and s < anterior:
            total += anterior
        anterior = s
    return total + (anterior or 0.0)


def velocidade_atual(historico, janela=20):
    """Segundos por iteração nas últimas iterações da sessão atual."""
    trecho = [historico[-1]]
    for r in reversed(historico[:-1]):
        if r['seconds'] > trecho[-1]['seconds'] or len(trecho) >= janela:
            break
        trecho.append(r)
    if len(trecho) < 2:
        return None
    return (trecho[0]['seconds'] - trecho[-1]['seconds']) / (trecho[0]['iteration'] - trecho[-1]['iteration'])


def fator(flags):
    """Custo relativo do trial: horizonte × episódios, em relação a 192 × 3."""
    def valor(nome, padrao):
        return float(flags[flags.index(nome) + 1]) if nome in flags else padrao
    return valor('--horizon', 192) * valor('--episodes', 3) / (192 * 3)


def situacao(pasta):
    """-> ('completo', duração) | ('rodando', dict) | ('fila', None)."""
    execucoes = sorted(pasta.glob('*/training.json'))
    for t in reversed(execucoes):
        if (ler_json(t.parent / 'status.json', {}) or {}).get('training_complete'):
            return 'completo', tempo_treinado(ler_json(t, []))
    if not execucoes:
        return 'fila', None
    t = execucoes[-1]
    historico = ler_json(t, [])
    cfg = ler_json(t.parent / 'config.json', {})
    if not historico:
        return 'rodando', dict(iteracao=0, total=cfg.get('iterations', 300), seg_it=None)
    return 'rodando', dict(iteracao=historico[-1]['iteration'], total=cfg.get('iterations', 300),
                           seg_it=velocidade_atual(historico))


def formatar(segundos):
    if segundos is None:
        return '?'
    horas, resto = divmod(int(segundos), 3600)
    return f'{horas} h {resto // 60:02d} min' if horas else f'{resto // 60} min'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fase', default='B', choices=['A', 'B', 'C'])
    parser.add_argument('--sementes', default='0,1,2')
    parser.add_argument('--paralelo', type=int, default=4)
    parser.add_argument('--base', default='resultados')
    args = parser.parse_args()
    base = Path(args.base)
    sementes = [int(s) for s in args.sementes.split(',')]
    if args.fase == 'A':
        tabela = {i: f for i, (_, _, f) in exp.FASE_A.items()}
    elif args.fase == 'B':
        tabela = {i: f for i, (_, _, f) in exp.FASE_B.items()}
    else:
        tabela = {'C-MELHOR': [], 'C-SEM': []}

    completos, rodando, fila = [], [], []
    for ident, flags in tabela.items():
        for s in sementes:
            estado, info = situacao(base / ident / f's{s}')
            item = dict(id=ident, semente=s, fator=fator(flags), info=info)
            {'completo': completos, 'rodando': rodando, 'fila': fila}[estado].append(item)

    # Duração de uma execução "padrão" (trial de 192 × 3), a partir do que já se sabe.
    amostras = [c['info'] / c['fator'] for c in completos if c['info']]
    amostras += [r['info']['seg_it'] * r['info']['total'] / r['fator'] for r in rodando if r['info']['seg_it']]
    padrao = sum(amostras) / len(amostras) if amostras else None

    print(f'Fase {args.fase}: {len(completos) + len(rodando) + len(fila)} execuções — '
          f'{len(completos)} completas, {len(rodando)} rodando, {len(fila)} na fila\n')
    restante = 0.0
    if rodando:
        print('Rodando agora:')
    for r in rodando:
        i = r['info']
        if i['seg_it']:
            falta = (i['total'] - i['iteracao']) * i['seg_it'] + AVALIACAO_S
        else:
            falta = padrao * r['fator'] + AVALIACAO_S if padrao else None
        restante += falta or 0
        vel = f'{i["seg_it"]:.0f} s/iteração' if i['seg_it'] else 'começando'
        print(f'  {r["id"]:7s} s{r["semente"]}  iteração {i["iteracao"]:3d}/{i["total"]}  {vel:16s} falta ~{formatar(falta)}')
    if fila:
        print('\nNa fila: ' + ', '.join(f'{f["id"]} s{f["semente"]}' for f in fila))
        if padrao:
            restante += sum(padrao * f['fator'] + AVALIACAO_S for f in fila)
    if not rodando and not fila:
        print('Fase concluída.')
        return
    if padrao or not fila:
        estimativa = restante / max(1, min(args.paralelo, len(rodando) + len(fila)))
        print(f'\nEstimativa para terminar a fase: ~{formatar(estimativa)} '
              f'(com {args.paralelo} em paralelo; aproximada, melhora conforme mais execuções terminam)')
    else:
        print('\nAinda sem dados para estimar; rode de novo em alguns minutos.')


if __name__ == '__main__':
    main()
