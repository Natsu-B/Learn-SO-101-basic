// Physical AI 応用編1 最終課題 説明資料
// 組版: typst compile docs/report/report.typ  (Typst 0.15)
// 数値の出典は docs/PICK_PLACE.md, docs/OBJECT_PHYSICS.md,
// docs/PHYSICS_AUDIT.md, docs/FORCE_CONTROL.md と runs/ の評価ログ。

#let account = "omnicampusアカウント名"
#let repo = "https://github.com/Natsu-B/Learn-SO-101-basic"
#let video = "（YouTubeのURL）"

#set document(title: "SO-101による位置可変Pick & Placeと、柔らかさの分からない物体の把持力制御")
#set page(paper: "a4", margin: (x: 19mm, top: 17mm, bottom: 17mm), numbering: "1 / 1")
#set text(font: ("Libertinus Serif", "Hiragino Mincho ProN"), size: 9.5pt, lang: "ja")
#set par(justify: true, leading: 0.76em, first-line-indent: (amount: 1em, all: true), spacing: 0.9em)
#show heading: set text(font: ("Hiragino Kaku Gothic ProN"), weight: "bold")
#show heading.where(level: 1): it => block(above: 1.25em, below: 0.65em, text(size: 12.5pt, it))
#show heading.where(level: 2): it => block(above: 0.95em, below: 0.5em, text(size: 10.5pt, it))
#set heading(numbering: "1.1")
#show raw: set text(font: ("Menlo", "Hiragino Kaku Gothic ProN"), size: 7.8pt)
#show figure.caption: set text(size: 8.5pt)
#show figure: set block(above: 0.9em, below: 0.9em)
#set table(stroke: (x, y) => if y == 0 { (bottom: 0.6pt) } else { none }, inset: (x: 5pt, y: 3.1pt))
#show table: set text(size: 8.3pt)
#show table.cell.where(y: 0): set text(font: ("Hiragino Kaku Gothic ProN"), weight: "bold")
#show link: set text(fill: rgb("#1c5cab"))

#let box-node(title, body, fill: rgb("#eef4fc")) = rect(
  width: 100%, inset: 6pt, radius: 3pt, fill: fill, stroke: 0.6pt + rgb("#9bb7da"),
  [#text(font: "Hiragino Kaku Gothic ProN", weight: "bold", size: 8.6pt, title) \ #text(size: 8pt, body)],
)

// ---------------------------------------------------------------- 表題
#align(center)[
  #text(font: "Hiragino Kaku Gothic ProN", size: 15pt, weight: "bold")[
    SO-101による位置可変Pick & Placeと、\ 柔らかさの分からない物体の把持力制御
  ]
  #v(0.3em)
  #text(size: 9pt)[Physical AI 応用編1 最終課題 説明資料　#account]
  #v(0.1em)
  #text(size: 8.6pt)[コード: #link(repo)[#repo]　　動画: #video（約2分、すべて等倍）]
]
#v(0.5em)

= 概要

MuJoCo上のSO-101アームに、二つのタスクを実行させた。
一つ目は課題の本体で、テーブル上の25 mm立方体を掴み、目標位置まで運んで置く。
立方体の初期位置と目標位置は試行ごとに乱数で変わる。
掴む動作と置く動作は逆運動学（IK）で制御し、運ぶ動作は強化学習で得た方策（観測から関節の指令を出すニューラルネットワーク）の推論で制御した。
初期位置の異なる200試行すべてで配置に成功し、置いた位置の誤差は平均2.42 mm、最大6.42 mmだった。

二つ目は独自に追加した課題で、柔らかさと摩擦と質量が分からない物体を、指定した力で潰さずに掴む。
触覚による把持力制御を強化学習で学習した論文 @lach2024 を調べ、SO-101に移植した。
その過程で、SO-101の片側しか動かないグリッパだけでは、物体を固定側のジョーまで押し寄せる間に柔らかい物体が潰れることが分かった。
そこで腕の横移動とグリッパを同時に動かす方策を学習させ、潰さずに目標力で掴めた試行の割合を、グリッパだけの最良の制御の19.0 %から92.0 %に上げた。

工夫した点は二つある。
一つは、動作ごとの成功判定を一か所で定義し、学習と評価と動画で同じ判定を使ったことである。
もう一つは、論文の手法がSO-101で失敗する原因を計測で突き止め、腕の微小移動で補う方法に作り替えたことである。
数値はすべて、リポジトリにある評価スクリプトの出力から書いた。
実機は使っていない。

= タスクとシステムの構成

== シミュレータとソフトウェア

物理計算とタスク管理はRustで書き、強化学習だけをPythonのStable-Baselines3（以下SB3）で行った（@fig-arch）。
Rust側はMuJoCoのモデル読み込み、接触の判定、IK、Skill（掴む、運ぶ、置くの各動作）、成功判定を持つ。
PythonとはPyO3で接続し、Gymnasium環境としてSB3に渡す。
制御周期は30 Hz、物理計算の刻みは5 msで、SO-101モデル（MuJoCo Menagerie）の関節範囲とアクチュエータの力の上限はそのまま使った。
立方体の位置をプログラムから書き込むのはリセット時だけで、その後は接触の物理だけで動く。

#figure(
  grid(columns: (1fr, 0.5fr, 1fr), column-gutter: 6pt, align: horizon,
    box-node("Rust（mujoco-runtime）",
      [シーンと物体のランダム化 \ 接触の判定と5 msごとの計測 \ IKとSkill（Grasp / Transport / Place）\ 成功判定（唯一の定義）]),
    align(center, text(size: 8pt)[PyO3 \ #sym.arrow.l.r]),
    box-node("Python（robot_sim）",
      [Gymnasium環境 \ SAC+HER（Transport）\ PPO（把持力制御）\ 評価と動画の書き出し], fill: rgb("#fdf0ea")),
  ),
  caption: [システムの構成。成功判定はRust側で一度だけ定義し、学習、評価、動画で共用する。],
) <fig-arch>

== Pick & Placeの手順と成功判定

立方体（25 mm、30 g）と目標は、どちらもアーム根元から半径0.15〜0.25 m、方位±0.8 radの範囲に、互いに0.08 m以上離して置く。
立方体の向き（yaw）は0に固定した。
制御にはシミュレータから得た立方体と目標の位置の真値を使い、画像認識は扱っていない。
一つの試行はGrasp、Transport、Placeの三つのSkillを順に実行し、前のSkillの成功判定が通ったら次へ進む。
前のSkillの終了状態が次のSkillの初期状態になるので、Transportの学習でも、実際にGraspを実行した後の状態から開始した。
各Skillの成功判定を @tab-criteria に示す。

#figure(
  table(columns: (auto, 1fr), align: (left, left),
    [Skill], [成功の条件],
    [Grasp（IK）], [両ジョーが立方体に接触し、テーブルから離れ、立方体中心が静止時より30 mm以上高い状態が0.3 s続く],
    [Transport（SAC）], [両ジョーで挟んだまま、立方体中心が静止時より15 mm以上高く、目標とのXY距離が15 mm以内],
    [Place（IK）], [ジョーが離れてテーブルに接し、高さの誤差3 mm未満、速さ0.01 m/s未満、目標とのXY距離20 mm以内],
  ),
  caption: [成功判定（Rustの`predicates`）。学習環境、評価、動画で同じ定義を使う。],
) <tab-criteria>

== 学習済みモデルを使った箇所

Transportは、SAC（連続値の行動を学習する強化学習手法）とHER @andrychowicz2017（失敗した経験の目標を、実際に到達した位置に置き換えて再利用する手法）で学習した方策を、毎制御周期に推論して動かす。
観測は腕5関節の角度と角速度、挟持中心と立方体の位置、立方体の速度、グリッパ角など23次元、行動は腕5関節の角度の増分で、グリッパは閉じたまま保つ。
報酬は立方体と目標の距離だけで決めた。
学習は5万stepで止め、学習中の評価で最も良かった3万step時点のモデルを使った（このモデルはTransport単体の評価で20/20成功）。
発展課題の方策は、別の強化学習手法であるPPO @schulman2017 で学習した（@sec-force）。

= 位置可変Pick & Placeの結果

seed 0〜199の200試行で評価した（@tab-pp）。
学習したTransportを使った構成は200試行すべてで成功し、立方体の落下は0件だった。
同じseed列で全SkillをIKにした構成も200/200で、最終誤差の平均はほぼ同じ（2.43 mm）である。
学習したTransportは、IKで書いた運搬と同等の結果を、学習済み方策の推論で出せた。
@fig-pp に動画の一場面を示す。

#figure(
  table(columns: (auto, auto, auto, auto, auto, auto), align: (left, center, center, center, center, center),
    [構成], [全体], [Grasp], [Transport], [Place], [最終XY誤差 平均 / P95 / 最大],
    [IK → *SAC（学習）* → IK], [200/200], [200/200], [200/200], [200/200], [2.42 / 3.43 / 6.42 mm],
    [すべてIK], [200/200], [200/200], [200/200], [200/200], [2.43 / 3.94 / 6.95 mm],
  ),
  caption: [Pick & Placeの評価（seed 0〜199、各200試行。落下はどちらも0件）。],
) <tab-pp>

#figure(
  grid(columns: 3, column-gutter: 4pt,
    image("figures/pick_place_demo_grasp.png"),
    image("figures/pick_place_demo_transport.png"),
    image("figures/pick_place_demo_place.png"),
  ),
  caption: [動画の場面（左からGrasp、学習したTransport、Place）。画面上部に、実行中のSkillと、学習済みモデルかIKかを表示した。],
) <fig-pp>

開発中に実際に起きた失敗と、その原因は次の二つだった。
IKの姿勢誤差に外積を使うと、指先が真上を向く解に収束したので、方向ベクトルの差に変えた。
固定ジョーの内面と立方体の側面のすき間は約1 mmしかなく、下降中に立方体の上面へ当たることがあった。
下降中だけ固定ジョーを4 mm離し、閉じる動作で立方体を押し付けるようにした。
最初はずらす向きを逆にしてしまい、成功が95/200まで落ちたので、ログで確かめて直した。

物理の確認はコンパイルと分けてテストにした。
両ジョーで挟めば30 mm以上持ち上がり、同じ動作でグリッパを閉じなければ持ち上がらない、という対照実験をRustのテストに入れてある。
動画の描画は物理状態をコピーして行い、描画の有無で軌道が変わらないこと（seed 0で完全一致）を確認した。

= 柔らかさの分からない物体の把持力制御 <sec-force>

== 取り組んだ理由と実験の条件

本体のPick & Placeでは、グリッパに−0.17 radの閉指令を与え続けるため、把持中のアクチュエータのトルクが上限の2.94 N·mに達し、立方体を片側約39 Nで握っていた。
重さ30 gの立方体に対して大きな力で、柔らかい物体を扱うなら、物体が耐えられる力に合わせて握る力を調整する必要がある。
未知の質量と摩擦に対して掴み方を学習させる実験（残差強化学習）も行ったが、IKや固定の補正に勝てなかった（@sec-negative）。
そこで、握る力そのものを接触力のフィードバックで制御する実験に切り替えた。

この実験は、Pick & Placeとは別の独立した環境で行った。
アームを物体の真上に下ろした状態から始め、テーブル上の模擬物体を5 s（150制御周期）掴む。
物体は持ち上げない。
目標力は試行ごとに、物体が耐えられる力（剛性×最大圧縮量）の80 %以下かつ8 N以下の範囲から選び、観測として方策に与える。
物体の剛性（200〜5000 N/m）、幅、ジョーとのすき間、摩擦、質量も試行ごとに変えるが、こちらは方策に与えない。

== 論文の調査と、柔らかい物体のモデル

参考にしたのは、TIAGoの2指グリッパで、触覚センサの力を見ながらPPOで把持力を制御した研究 @lach2024 である。
調査と再実装には、公開されているarXiv初版（2311.07245v1、2023年）を用いた。
公開実装にはライセンスの表記がなかったので、コードは写さず、論文の式と設定値をもとに実装し直した。
論文と公開コードには食い違いがあり、たとえば論文の主な手法は行動を接触状態に応じて縮める仕組み（帰納バイアス）を使うが、公開の学習スクリプトではこれが無効で、物体の移動に対する罰の重みも0だった。
論文の柔らかさは、MuJoCoの数値的なめり込みとセンサ値の係数で表されていて、物体の変形状態はモデルに存在しない。

この点を変え、物体は剛体の芯の4側面に、ばねとダンパ付きのスライド関節で板を付けたモデルにした（@fig-fgdemo の青い部分）。
板の押し込み量を $d$、押し込み速度を $dot(d)$、ばね定数を $k$、減衰係数を $c$ とすると、板の復元力は $F = k d + c dot(d)$ になる。
押し込みが上限に達した状態を潰れ、ジョーが板に1 mmを超えてめり込んだ状態を無効な接触として、変形とは別に記録した。
静的な保持試験では、潰れる前のジョーの法線力とばねとダンパによる力の差が0.05 N以内だった。
接触力の最大値は、30 Hzの制御周期の終わりだけでなく、5 msの物理ステップごとに記録した。
衝突の瞬間の力は周期の終わりの値の2倍以上になることがあり、周期の終わりだけを見ると見落とす。

成功は、両ジョーの接触を獲得し、試行中に潰れと無効な接触がなく、最後の1 sで接触を失わず、その間の平均力の誤差が目標力の10 %と0.05 Nの大きい方以内に収まった試行とした。

== グリッパだけでは潰れる理由

論文の公開学習スクリプトに合わせ、物体移動の罰と帰納バイアスを外したPPOは、力の平均追従誤差が0.208 Nで、SO-101向けに調整した力PI制御（論文のベースラインと同じ構造）の0.278 Nを下回った。
論文どおりに罰と帰納バイアスを入れた設定では、誤差は2.70 Nだった。
どちらの設定でも、潰さずに目標力で掴めた試行は15 %前後で、力PIの16.5 %と変わらなかった（@fig-force）。

失敗を段階に分けて数えると、潰れの多くは目標力に届く前、可動ジョーが物体を固定ジョーまで押し寄せる間に起きていた。
SO-101は片側のジョーしか動かないので、物体が可動ジョー側に寄っていれば、テーブル上を押して運ぶしかない。
この押し寄せに必要な力を測った。
可動ジョーをゆっくり閉じても、物体が滑り出すときの力はテーブル摩擦からの見積もりの中央値1.72倍あり、22 %の物体では板が耐えられる力を超えた。
回転するジョーが物体を斜め下にも押し、摩擦を増やしているためと考えている。
ゆっくり押しても36 %の試行で板が潰れたので、グリッパの制御だけでは避けきれない失敗と判断した。
報酬を変えた学習（ピーク力の罰、両ジョー接触の獲得に対する報酬）や、押す力の上限を目標力の0.8倍から毎秒0.5 Nずつ上げるスクリプト制御（二段階PI）も試したが、成功率は最大19 %だった。

== 腕とグリッパの協調

論文の2本指では、先に触れた指を止め、もう一方の指を寄せれば物体を押さずに済む。
SO-101では、腕を把持方向へ数mm動かせば、固定ジョーを物体に寄せられる。
そこで腕の目標位置を把持方向へ最大12 mmずらす操作をRustに追加し、まずスクリプト制御（腕補正PI）で試すと、成功率は44.0 %になった。

次に、グリッパの指令と腕の移動量を同時に出す2次元の方策をPPOで学習させた。
報酬は、両ジョーが接触している間だけ力の追従を評価し、両ジョーの接触の獲得に+20、試行の最後の安全把持の成功に+50、潰れと無効な接触にそれぞれ−10を与えた。
一度も両ジョーで接触しなければ−20とし、接触を避けて罰を逃れる方策が得をしないようにした。
50万stepで成功率67.5 %になったので、累積約100万stepまで延長した。
選択用の48試行で最良だった約70万step時点のモデルを、評価用の別の200試行で評価すると92.0 %だった（@fig-force、@tab-force、@fig-fgdemo）。
同じ200試行で成否を対応させると、腕補正PIより成功した試行が100件、逆が4件で、McNemar検定で $p = 4.7 times 10^(-25)$ だった。
腕だけを学習し、握る力はPIに任せた構成でも72.0 %になり、改善の多くは腕を動かす時機と速さの学習から来ている。

#figure(
  image("figures/force_grasp_comparison.png", width: 100%),
  caption: [潰さずに目標力で掴めた割合（安全把持の成功率）と潰れ率。評価用の200試行（seed 8,000,000〜8,000,199）で、全制御が同じ物体、目標力、センサノイズを使う。PPO P9は両ジョー接触の獲得を報酬に加えたグリッパだけの方策、P10とP10bは腕とグリッパの2次元の方策（50万stepと延長後）、P11は腕だけを学習しグリッパはPIの構成。],
) <fig-force>

#figure(
  table(columns: (auto, auto, auto, auto, auto, auto), align: (left, center, center, center, center, center),
    [制御（200試行）], [成功], [潰れ], [無効な接触], [物体の移動], [ピーク力 中央値 / P95],
    [力PI（グリッパのみ）], [16.5 %], [66.5 %], [30.0 %], [7.3 mm], [14.4 / 33.7 N],
    [PPO P9（グリッパのみ）], [19.0 %], [66.0 %], [34.5 %], [7.7 mm], [11.4 / 45.2 N],
    [腕補正PI（スクリプト）], [44.0 %], [20.0 %], [21.5 %], [1.8 mm], [2.5 / 8.4 N],
    [PPO P10b（腕＋グリッパ）], [*92.0 %*], [1.5 %], [5.5 %], [1.1 mm], [2.2 / 8.2 N],
  ),
  caption: [把持力制御の比較。ピーク力は、両ジョーで接触し、無効な接触がなかった試行だけで集計した。],
) <tab-force>

#figure(
  grid(columns: 2, column-gutter: 4pt,
    image("figures/force_grasp_demo_8000091_pi.png"),
    image("figures/force_grasp_demo_8000091_P10b.png"),
  ),
  caption: [同じ柔らかい物体（剛性306 N/m、目標力0.68 N）での比較。左の力PIは押し寄せる間に物体を潰し（最大28.7 N）、右の腕とグリッパのPPOは潰さずに掴んだ（最大2.4 N）。],
) <fig-fgdemo>

選択用の成功率はチェックポイントごとに大きく揺れ、延長した学習では0.15〜0.90の間を動いた（@fig-curves）。
同じ学習の最後のモデルは評価用の200試行で78.5 %で、どのチェックポイントを選ぶかで成績が変わる。
学習はまだ安定していない。

#figure(
  image("figures/force_grasp_coop_curves.png", width: 88%),
  caption: [腕とグリッパの協調学習の推移（選択用の48試行）。P10は50万step、P10bはP10を延長した学習、P11は腕だけを学習した構成。],
) <fig-curves>

= うまくいかなかったことと限界 <sec-negative>

未知の質量と摩擦に対して、IKのGraspに学習した補正を足す実験を行った。
質量20〜100 g、摩擦係数0.05〜1.5で学習し、学習範囲外の120試行を加えた計570試行で評価した。
全工程の成功数は、最初の報酬で192〜237、後段の成功を含めた報酬で222〜235で、IKのみの240と手首の固定補正の253を下回った。
この比較は、摩擦を変えると接触の次元と硬さも変わる最初の接触環境で行った。
接触物理を調べると、軽い物体ほど指先が深くめり込むなど、MuJoCoの接触モデルに由来する偏りがあった。
手首の固定補正も、値を変えたときの差は並べ替え検定で偶然のばらつきと区別できなかった。

把持力制御では、論文どおりの設定がそのままでは働かなかった。
帰納バイアスを1自由度に合わせると、物体を押し寄せる間の閉じる速さが10分の1になり、150制御周期の間に両ジョーが触れなくなった。
ピーク力に罰を与えた設定では、物体に触れない行動が学習された。
両ジョー接触の獲得を報酬の達成条件にしたのは、この失敗を受けてのことである。

残っている限界は次のとおりである。
力の観測はシミュレーション上の接触力で、実機のSO-101には指先の力センサがない。
実機で使うには、センサを付けるか、サーボの電流から力を推定する必要がある。
硬い物体では衝突時に板がジョーへめり込む無効な接触が5.5 %残り、これは物理モデル側の問題である。
学習は乱数seed 1つだけで行い、把持力制御はPick & Placeの本体にはまだ組み込んでいない。

= 再現方法とAIの利用

すべてNixの開発環境（`nix develop`）で動く。
学習済みモデルは`runs/`に置き、Gitでは管理していない。
評価にはモデルと同じディレクトリの`config.json`が必要なので、下の手順で学習して用意する。
92.0 %のモデルは、下の50万stepの学習の最後のモデルから`--resume`で50万step追加学習したもので、延長学習と全比較の評価コマンドは`docs/FORCE_CONTROL.md`の§11に記載した。

```bash
cargo test --workspace && maturin develop --release && pytest -q      # Rust 66件 / Python 181件
python examples/python/train_transport_sac_her.py --steps 250000 --seed 42 \
  --output runs/so101-transport/seed-42                                 # 5万stepで停止して使用
python examples/python/eval_pick_place.py --model runs/so101-transport/seed-42/best.zip \
  --episodes 200 --seed 0 --transport learned                          # 表2
python examples/python/train_force_grasp_ppo.py --output runs/force-grasp-coop/p10 --steps 500000 \
  --eval-freq 49152 --eval-episodes 48 --checkpoint-freq 98304 --no-bias --alpha-obj 0 0 \
  --reward coop_safe --action-mode arm_gripper --select success
python examples/python/eval_force_grasp.py --output runs/eval --episodes 200 --controllers pi \
  armpi:push_ramp=0.5,push_speed=0.3,push_ratio=0.8,shift_speed=0.012 \
  ppo:P10=runs/force-grasp-coop/p10/best.zip
```

課題の要件であるAIの利用は、学習済み方策の推論である。
Pick & PlaceのTransportはSAC+HERの方策、把持力制御はPPOの方策が、毎制御周期に行動を出す。
開発では、Claude CodeとCodexをコードの作成、実験の実行、文書の作成と点検に使った。
表の数値は評価スクリプトの出力から転記し、評価条件とログの保存先をリポジトリの`docs/`に記録した。

#set text(lang: "en")
#bibliography("refs.yml", title: "参考文献", style: "ieee")
