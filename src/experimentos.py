"""Plano de experimentos do RL² no Crafter, organizado por hipóteses.

Fases:
    A  recompensa      (H1, H2)        6 configurações × sementes 0 e 1
    B  hiperparâmetros (H3–H7)         8 variações da recompensa de referência × sementes 0 e 1
    C  confirmação     (H6 e efeito)   melhor configuração e seu controle sem memória × sementes 2, 3 e 4

Regras de passagem (decididas antes de ver o teste):
    referência para B = grupo da fase A com maior validação (ranking do analisar.py);
    melhor para C     = variação da fase B (ou a própria referência) com maior validação, sem contar B-SEM.

Uso:
    python experimentos.py listar [--base resultados]
    python experimentos.py rodar --fase A [--paralelo 2]
    python experimentos.py rodar --fase B --referencia A-R2
    python experimentos.py rodar --fase C --referencia A-R2 --melhor B-H96
    python experimentos.py comandos --fase A          (só mostra os comandos)
    Opções: --ids A-R2,A-R3  --sementes 0,1  --device cpu  --max-minutes 600  --smoke

Rodar de novo o mesmo comando é seguro: execuções completas são puladas e as
interrompidas (sessão do Colab caiu, limite de tempo) são retomadas do último
last.pt. Mantenha o mesmo --device para retomar. Cada execução fica em
BASE/<id>/s<semente>/<data_hora>/, com experimento.json em BASE/<id>/s<semente>/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

AQUI = Path(__file__).resolve().parent
TRIAGEM = ['--envs', '16', '--iterations', '300', '--validate-every', '10', '--test-repeats', '16']
CONFIRMACAO = ['--envs', '16', '--iterations', '600', '--validate-every', '20', '--test-repeats', '25', '--video']

HIPOTESES = {
    'H1': 'Recompensas repetíveis levam a uma estratégia fixa ("beber sempre"); limitar quantos '
          'eventos contam por episódio a elimina.',
    'H2': 'Sem custo para objetivos errados, "fazer tudo" dispensa identificar o objetivo; penalizar '
          'objetivos errados aumenta a seletividade e o efeito da memória, até o agente ficar passivo.',
    'H3': 'Mais crédito entre episódios (γ maior) aumenta o efeito da memória.',
    'H4': 'Episódios mais curtos tornam identificar o objetivo mais valioso e aumentam o efeito da memória.',
    'H5': 'Mais episódios por trial tornam a identificação mais valiosa e aumentam o efeito nos episódios finais.',
    'H6': 'A memória é necessária: o agente com GRU supera o mesmo agente treinado sem memória a partir do '
          'episódio 2, e empata no episódio 1.',
    'H7': 'Mais entropia (exploração) ajuda a identificar o objetivo no episódio 1. Exploratória.',
}

# id: (hipótese, descrição, flags)
FASE_A = {
    'A-R0': ('H1', 'original: +1 por evento, sem teto', ['--reward-cap', '0']),
    'A-R1': ('H1', 'teto de 3 eventos por episódio', ['--reward-cap', '3']),
    'A-R2': ('H1/H2', 'só o 1º evento de cada episódio', ['--reward-cap', '1']),
    'A-R3': ('H2', 'só o 1º evento + penalidade 0,1 por objetivo errado', ['--reward-cap', '1', '--wrong-penalty', '0.1']),
    'A-R4': ('H2', 'só o 1º evento + penalidade 0,5 por objetivo errado', ['--reward-cap', '1', '--wrong-penalty', '0.5']),
    'A-R5': ('H1×H2', 'sem teto + penalidade 0,1 (a penalidade sozinha resolve?)', ['--reward-cap', '0', '--wrong-penalty', '0.1']),
}
FASE_B = {
    'B-G995': ('H3', 'γ = 0,995', ['--gamma', '0.995']),
    'B-G999': ('H3', 'γ = 0,999', ['--gamma', '0.999']),
    'B-H96': ('H4', 'episódios de 96 passos', ['--horizon', '96']),
    'B-K2': ('H5', '2 episódios por trial', ['--episodes', '2']),
    'B-K5': ('H5', '5 episódios por trial', ['--episodes', '5']),
    'B-SEM': ('H6', 'controle: sem memória', ['--no-memory']),
    'B-E03': ('H7', 'entropia 0,03', ['--entropy-coef', '0.03']),
    'B-E003': ('H7', 'entropia 0,003', ['--entropy-coef', '0.003']),
}
SEMENTES = {'A': [0, 1], 'B': [0, 1], 'C': [2, 3, 4]}
_TRAVA = threading.Lock()


def avisar(mensagem):
    """print que não mistura linhas de execuções em paralelo."""
    with _TRAVA:
        print(mensagem, flush=True)


def experimentos_da_fase(fase, referencia='', melhor=''):
    """-> {id: dict(hipotese, descricao, flags)} com as flags completas de cada execução."""
    if fase == 'A':
        return {i: dict(hipotese=h, descricao=d, flags=TRIAGEM + f) for i, (h, d, f) in FASE_A.items()}
    if referencia not in FASE_A:
        raise SystemExit(f'A fase {fase} precisa de --referencia com um id da fase A ({", ".join(FASE_A)}).')
    base_ref = FASE_A[referencia][2]
    if fase == 'B':
        return {i: dict(hipotese=h, descricao=f'{d} (recompensa de {referencia})', flags=TRIAGEM + base_ref + f)
                for i, (h, d, f) in FASE_B.items()}
    if fase == 'C':
        if melhor not in FASE_B and melhor != 'ref':
            raise SystemExit(f'A fase C precisa de --melhor com um id da fase B ou "ref" ({", ".join(FASE_B)}).')
        if melhor == 'B-SEM':
            raise SystemExit('B-SEM é o controle sem memória; escolha a melhor variação com memória.')
        extra = [] if melhor == 'ref' else FASE_B[melhor][2]
        nome = referencia if melhor == 'ref' else f'{referencia} + {melhor}'
        return {
            'C-MELHOR': dict(hipotese='efeito', descricao=f'confirmação: {nome}', flags=CONFIRMACAO + base_ref + extra),
            'C-SEM': dict(hipotese='H6', descricao=f'confirmação sem memória: {nome}',
                          flags=CONFIRMACAO + base_ref + extra + ['--no-memory']),
        }
    raise SystemExit('Fase deve ser A, B ou C.')


def situacao(pasta):
    """-> ('completo', pasta_da_execucao) | ('parcial', last.pt) | ('novo', None)."""
    for status in sorted(pasta.glob('*/status.json'), reverse=True):
        try:
            s = json.loads(status.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if s.get('status') == 'completed' and s.get('training_complete'):
            return 'completo', status.parent
    ultimos = sorted(pasta.glob('*/last.pt'))
    return ('parcial', ultimos[-1]) if ultimos else ('novo', None)


def montar_comando(args, flags, semente, pasta, retomar=None):
    cmd = [sys.executable, str(args.script), *flags, '--seed', str(semente), '--device', args.device,
           '--max-minutes', str(args.max_minutes), '--output', str(pasta)]
    if retomar:
        cmd += ['--resume', str(retomar)]
    if args.smoke:
        cmd += ['--smoke']
    return cmd


def diagnosticar(args, execucao, log):
    if (execucao / 'comportamento_resumo.json').exists():
        return
    cmd = [sys.executable, str(args.diagnostico), '--script', str(args.script),
           '--checkpoint', str(execucao / 'best.pt'), '--repeats', '1' if args.smoke else '8', '--envs', '8']
    with log.open('a', encoding='utf-8') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)


def rodar_um(args, ident, info, semente):
    pasta = args.base / ident / f's{semente}'
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / 'experimento.json').write_text(json.dumps(dict(
        id=ident, fase=args.fase, hipotese=info['hipotese'], descricao=info['descricao'], flags=info['flags'],
        semente=semente, referencia=args.referencia or None, melhor=args.melhor or None), indent=2,
        ensure_ascii=False), encoding='utf-8')
    log = pasta / 'log.txt'
    estado, alvo = situacao(pasta)
    rotulo = f'{ident} s{semente}'
    if estado == 'completo':
        diagnosticar(args, alvo, log)
        avisar(f'[pulado]   {rotulo}: já completo')
        return True
    inicio = time.monotonic()
    avisar(f'[iniciado] {rotulo}' + (' (retomando)' if estado == 'parcial' else ''))
    with log.open('a', encoding='utf-8') as f:
        f.write(f'\n=== {time.strftime("%Y-%m-%d %H:%M:%S")} {rotulo} ===\n')
        f.flush()
        codigo = subprocess.run(montar_comando(args, info['flags'], semente, pasta,
                                               alvo if estado == 'parcial' else None),
                                stdout=f, stderr=subprocess.STDOUT, check=False).returncode
    minutos = (time.monotonic() - inicio) / 60
    estado, alvo = situacao(pasta)
    if codigo == 0 and estado == 'completo':
        diagnosticar(args, alvo, log)
        avisar(f'[completo] {rotulo} em {minutos:.0f} min')
        return True
    if codigo == 0:
        avisar(f'[parcial]  {rotulo}: limite de tempo atingido após {minutos:.0f} min; rode de novo para continuar')
    else:
        avisar(f'[falhou]   {rotulo} (código {codigo}); veja {log}')
    return False


def selecionar(args):
    experimentos = experimentos_da_fase(args.fase, args.referencia, args.melhor)
    ids = args.ids.split(',') if args.ids else list(experimentos)
    desconhecidos = [i for i in ids if i not in experimentos]
    if desconhecidos:
        raise SystemExit(f'Ids desconhecidos na fase {args.fase}: {", ".join(desconhecidos)}')
    sementes = [int(s) for s in args.sementes.split(',')] if args.sementes else SEMENTES[args.fase]
    # Todas as configurações da semente 0 antes da semente 1: se o tempo acabar, há ao menos uma de cada.
    return [(i, experimentos[i], s) for s in sementes for i in ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('acao', choices=['listar', 'rodar', 'comandos'])
    parser.add_argument('--fase', choices=['A', 'B', 'C'], default='A')
    parser.add_argument('--referencia', default='', help='id da fase A usado como recompensa nas fases B e C')
    parser.add_argument('--melhor', default='', help='id da fase B (ou "ref") confirmado na fase C')
    parser.add_argument('--ids', default='', help='só estes ids, separados por vírgula')
    parser.add_argument('--sementes', default='', help='sementes separadas por vírgula (padrão: A/B 0,1; C 2,3,4)')
    parser.add_argument('--paralelo', type=int, default=1, help='execuções ao mesmo tempo (no máximo núcleos − 1)')
    parser.add_argument('--base', default='', help='pasta dos resultados (padrão: resultados, ou resultados_smoke)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--max-minutes', type=float, default=600, help='limite por execução; rode de novo para continuar')
    parser.add_argument('--smoke', action='store_true', help='tudo com orçamento mínimo, para testar o encadeamento')
    parser.add_argument('--script', default=str(AQUI / 'crafter_meta_goals_rl2_ppo_v3.py'))
    parser.add_argument('--diagnostico', default=str(AQUI / 'diagnostico_comportamento.py'))
    args = parser.parse_args()
    args.base = Path(args.base or ('resultados_smoke' if args.smoke else 'resultados')).resolve()
    args.script, args.diagnostico = Path(args.script).resolve(), Path(args.diagnostico).resolve()

    if args.acao == 'listar':
        print('Hipóteses:')
        for h, texto in HIPOTESES.items():
            print(f'  {h}: {texto}')
        marcas = {'completo': '✓', 'parcial': '~', 'novo': '·'}
        for fase, tabela in (('A', FASE_A), ('B', FASE_B)):
            print(f'\nFase {fase} (sementes {SEMENTES[fase]}):')
            for ident, (h, d, f) in tabela.items():
                estados = ' '.join(f's{s}{marcas[situacao(args.base / ident / f"s{s}")[0]]}' for s in SEMENTES[fase])
                print(f'  {ident:7s} {h:6s} {d:58s} {" ".join(f):38s} {estados}')
        print('\nFase C (sementes [2, 3, 4]): C-MELHOR e C-SEM, montadas a partir de --referencia e --melhor.')
        print('Legenda: ✓ completo  ~ parcial (rode de novo para continuar)  · não iniciado')
        return
    trabalhos = selecionar(args)
    if args.acao == 'comandos':
        for ident, info, semente in trabalhos:
            pasta = args.base / ident / f's{semente}'
            print(' '.join(montar_comando(args, info['flags'], semente, pasta)))
        return
    print(f'Fase {args.fase}: {len(trabalhos)} execuções, {args.paralelo} por vez, resultados em {args.base}', flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.paralelo)) as pool:
        resultados = list(pool.map(lambda t: rodar_um(args, *t), trabalhos))
    print(f'{sum(resultados)} de {len(resultados)} execuções completas.', flush=True)
    if not all(resultados):
        print('Rode o mesmo comando de novo para retomar as que faltam.', flush=True)


if __name__ == '__main__':
    main()
