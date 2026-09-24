


## Challenging Tag

| 태그 이름                    | 태그 종류                                  | Defined in Union14M [1] |
| --------------------------- | ----------------------------------------- | ----------------------- |
| Artistic                    | True / False                              | O                       |
| Curve                       | True / False                              | O                       |
| Complex Background          | True / False                              | O                       |
| Multi-Words                 | True / False                              | O                       |
| Incomplete                  | True / False                              | O                       |
| Multi-Oriented              | Horizontal / Verticall / Diagnal(Others)  | O                       |
<!-- | Font                        | Regular / Irregular                       |                         | -->
| Character size consistency  | Consistent / Varied                       |                         |
| Character style consistency | Consistent / Varied                       |                         |

[1] Q. Jiang, J. Wang, D. Peng, C. Liu, and L. Jin, "Revisiting Scene Text Recognition: A Data Perspective," in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, Oct. 2023, pp. 20543–20554. arXiv:2307.08723. DOI: [10.48550/arXiv.2307.08723](https://doi.org/10.48550/arXiv.2307.08723) ([open access PDF, CVF](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_Revisiting_Scene_Text_Recognition_A_Data_Perspective_ICCV_2023_paper.pdf))


## Identification tags
Beyond Union14M, we provide identification tags to support a finer-grained analysis of ambiguity.

* STR 평가는 2가지로 구분되어야 한다. 
* context와 관계없이 시각적으로 명확한 글자를 읽는 능력
* context에 의존해야 하는 글자를 읽는 능력
* 하지만 모호한 글자를 사람이 context를 통해 추론한 평가 셋은 조심히 봐야됨. 추론이 틀릴 수 있고, 문맥에 의존성이 커질 수 있음 -> 구분해서 봐야 함
* 따라서 모호한 글자를 와일드카드 처리해서 모호한 글자가 중간에 있어도 시각적으로 명확한 것은 잘 맞추는 지를 볼 필요가 있고
* 와일드 카드에 대해 사람이 추론한 것에 대해서도 평가해서 문맥을 통한 보정 실력도 체크할 필요가 있다.


### Visually Identifiable
- Clear: 모든 글자가 시각적으로 명확 (읽을 수 있음)
- Ambiguous: 모호한 글자가 있음 (시각적으로는 읽기 어려운 글자가 있음)

### Linguistically Identifiable
True: 모든 글자를 시각적 또는 언어적 컨텍스트를 이용하여 읽을 수 있음
False: 하나 이상의 글자를 시각적 또는 언어적 컨텍스트를 이용하더라도 읽을 수 없음
※ We suggest that all visually identifiable samples are linguistically identifiable when limited to Korean characters.

### Unidentifiable sample labeling
* Use wildcard "□"
* "안녕하□요" means that all characters are visually or linguistically identifiable, except for the fourth character '□'.

와일드카드는 글자의 종류 자체를 평가하지는 않지만 글자의 존재는 평가할 수 있다.
추론이 필요한 글자에 와일드 카드 처리를 하면 모델이 (언어 추론등이 도입되는 과정에서) 시각적으로 명확한 글자에 대한 성능이 어떻게 변하는지를 명확히 분석할 수 있다. 

![Text ambiguity taxonomy](../images/text_ambiguity_texanomy%20.jpg)


### 


## Validation
* 한 명의 레이블러가 레이블링 후 교차 검증
* AIHub에 학습된 SVTRv2와 SVTRv1으로 추론 후 하나라도 틀린 샘플들에 대해 저자가 완전 검수



글자의 품질 유형
Unknown
두 주 레이블러가 서로 다르거나
한 명이라도 모르겠다고 함
Context-readable but not visually clear
두 주 레이블러는 같은 글자를 씀
하지만 “반대로 보려는 레이블러”는 합리적인 다른 해석을 제시함
Visually clear
두 주 레이블러가 같은 글자를 씀
반대로 보려는 레이블러도 결국 같은 글자를 씀