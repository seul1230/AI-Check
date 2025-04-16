# 👨‍👩‍👧 AI-Check (아이췤) – 아이와 부모의 행복한 금융 생활을 위한 AI 솔루션

## ✨ 프로젝트 개요

> 자녀의 건강한 소비 습관 형성과 부모의 걱정 없는 금융 환경을 위한 **AI 기반 가정 금융 케어 서비스**

복잡해지는 금융 환경과 증가하는 디지털 위협 속에서, AI-Check는 **소비 교육과 보안 보호**를 함께 다루는 **가정용 AI 솔루션**을 목표로 합니다.


<br>

## 💡 프로젝트 의의

- 기록 기능을 넘어, 자녀의 소비 행동을 분석하고 학습 기회를 제공합니다
- AI가 실시간 피드백을 통해 소비 습관 개선을 유도합니다
- **보이스피싱 및 스미싱 탐지**를 통해 가족의 디지털 보안을 강화합니다
- 금융 교육과 실생활 보안을 함께 고려한 **가정용 AI 시스템**입니다

<br>

## 📌 주요 기능

### 기능 1. 자녀의 건강한 금융 생활

#### 📊 용돈 리포트

- 월별/카테고리별 지출 패턴 시각화
- 부모의 정기 용돈 지급 판단 기준 제공

![용돈인상요청](img/용돈인상요청청.gif)

#### ✍️ 자동 용돈 기입장

- 수입/지출 발생 시 자동 금액 입력
- 자녀는 상세 내용만 작성하면 되는 **간편한 기록 환경 제공**

![용돈기입장](img/용돈기입장.gif)


#### 🤖 용돈 협상 AI (엄마 AI)

- **살까 말까?**: 소비 패턴을 분석해 충동구매 여부에 대해 조언
- **추가 용돈 요청**: AI에게 설득 → 성공 시 부모에게 용돈 인상 요청 메시지(대화 요약) 전송, 실패 시 ‘거절’ 피드백 제공

> 설득 기준, 용돈 한도, 대화 스타일 등은 **부모가 자유롭게 커스터마이징 가능**

![엄마AI](https://github.com/user-attachments/assets/631dbe48-07c9-4b6b-9d84-0c3b02c001d0)


<br>

### 기능 2. 자녀의 걱정 없는 금융 생활

#### 📞 보이스피싱 탐지

- 통화 발생 시 자동 녹음 시작
- 딥페이크 음성 + 통화 스크립트 기반 AI 분석
- 의심 정황 포착 시 **자녀와 가족 모두에게 실시간 알림** 제공
- “난 안전해요” 실시간으로 주고 받음으로써 **납치형 보이스피싱 효과적으로 예방**

![보이스피싱싱](img/보이스피싱.gif)

#### 🔗 스미싱 방지

- 문자 내 URL 자동 분석
- 악성 URL 감지 시 **경고 알림 전송** 및 클릭 차단 유도

![스미싱](https://github.com/user-attachments/assets/811bb508-f43a-411c-b1a4-98d435fab96f)


<br>

### 기능 3. 부모의 걱정 없는 금융 생활

- 자녀의 통화 및 문자 기반 **이상 행위 탐지 시스템** 구축
- AI가 실시간으로 위험을 분석하고 **가족에게 즉시 공유**
- **가족 단위의 디지털 보안 체계 구현**


<br>

## 👷 구조도 및 시스템 아키텍처

![image](https://github.com/user-attachments/assets/3e75393b-1748-4834-9b3d-aedaeb5a1d68)

1. **사용자 웹뷰 요청 발생**
    - 사용자가 하이브리드 앱 내의 웹뷰를 통해 정적 콘텐츠(예: HTML, CSS, JS)나 API 호출을 HTTPS로 보냅니다.
2. **CloudFront의 HTTPS 종료 및 요청 라우팅**
    - CloudFront가 클라이언트와의 HTTPS 연결을 종료하여 암호화된 요청을 평문(또는 설정에 따라 재암호화된) 형태로 오리진에 전달합니다.
    - 요청 URL 경로에 따라 CloudFront는 두 가지 오리진으로 요청을 분기합니다.
    - **정적 콘텐츠**: S3 버킷이 오리진으로 지정되어 있으면, CloudFront는 S3에 요청을 전달합니다.
    - **API 요청**: 예를 들어, URL 경로가 /api/*로 시작하면, CloudFront는 API 서버인 Spring Cloud Gateway(예: 8080 포트)로 요청을 전달합니다.
3. **정적 콘텐츠 처리 (S3)**
    - CloudFront가 S3 버킷에서 요청한 정적 파일을 검색합니다.
    - 캐싱된 콘텐츠가 있다면 즉시 반환하고, 없으면 S3에서 파일을 받아온 후 클라이언트로 응답을 전달합니다.
4. **API 요청 처리 (Spring Cloud Gateway)**
    - CloudFront가 전달한 API 요청을 Spring Cloud Gateway가 수신합니다.
    - Spring Cloud Gateway는 요청에 대해 인증, 로깅, 라우팅 등의 기본 처리를 수행합니다.
    - 요청 URL이나 헤더 정보에 따라 해당 API 호출을 적절한 마이크로서비스(내부 비즈니스 로직 서버)로 전달합니다.
    - 마이크로서비스에서 요청에 따른 처리를 수행하고 응답을 생성하여 Spring Cloud Gateway로 반환합니다.
5. **응답 반환 및 최종 사용자 전달**
    - Spring Cloud Gateway는 API 응답을 CloudFront에 다시 전달합니다.
    - CloudFront는 HTTPS 연결을 통해 최종 사용자 웹뷰에 응답을 전달합니다.

<br>

## 🛠 기술 스택

### 🖥️ Backend & DevOps

![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Apache Kafka](https://img.shields.io/badge/Kafka-231F20?style=for-the-badge&logo=apache-kafka&logoColor=white)
![RabbitMQ](https://img.shields.io/badge/RabbitMQ-FF6600?style=for-the-badge&logo=rabbitmq&logoColor=white)
![Nginx](https://img.shields.io/badge/Nginx-009639?style=for-the-badge&logo=nginx&logoColor=white) <br>
![Spring Boot](https://img.shields.io/badge/Spring%20Boot-6DB33F?style=for-the-badge&logo=spring-boot&logoColor=white)
![Spring JPA](https://img.shields.io/badge/Spring%20JPA-6DB33F?style=for-the-badge&logo=spring&logoColor=white)
![Spring Security](https://img.shields.io/badge/Spring%20Security-6DB33F?style=for-the-badge&logo=spring-security&logoColor=white)
![Spring Cloud](https://img.shields.io/badge/Spring%20Cloud-6DB33F?style=for-the-badge&logo=spring&logoColor=white)
![Spring Batch](https://img.shields.io/badge/Spring%20Batch-6DB33F?style=for-the-badge&logo=spring&logoColor=white)



### 🤖 AI

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![TFLite](https://img.shields.io/badge/TFLite-FFCC00?style=for-the-badge&logo=tensorflow&logoColor=black)
![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)


### 🌐 Frontend

![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-06B6D4?style=for-the-badge&logo=tailwind-css&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=next.js&logoColor=white)


###  🐝 ️Database

![Redis](https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)
![ElasticSearch](https://img.shields.io/badge/ElasticSearch-005571?style=for-the-badge&logo=elasticsearch&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white)



### 🛠️ CI/CD & Collaboration

![Jenkins](https://img.shields.io/badge/Jenkins-D24939?style=for-the-badge&logo=jenkins&logoColor=white)
![Gitlab](https://img.shields.io/badge/Gitlab-000000?style=for-the-badge&logo=jenkins&logoColor=white)
![Figma](https://img.shields.io/badge/Figma-a259ff?style=for-the-badge&logo=figma&logoColor=white)
![Notion](https://img.shields.io/badge/Notion-000000?style=for-the-badge&logo=notion&logoColor=white)
![Jira](https://img.shields.io/badge/Jira-0052CC?style=for-the-badge&logo=jira&logoColor=white)
![Mattermost](https://img.shields.io/badge/Mattermost-0058CC?style=for-the-badge&logo=mattermost&logoColor=white)


<br>


<br>

## 👥 팀원

|                         조예슬                          |                          유선우                           |                          이승우                           |                          이시우                           |                        이정현                             |                           김혜빈                           |
| :-----------------------------------------------------: | :-------------------------------------------------------: | :-------------------------------------------------------: | :-------------------------------------------------------: | :-------------------------------------------------------: | :--------------------------------------------------------: |
| <img src="https://github.com/seul1230.png" width="100"> | <img src="https://github.com/BrokenFinger98.png" width="100"> | <img src="https://github.com/swoolee97.png" width="100"> | <img src="https://github.com/LEE-SIU.png" width="100"> | <img src="https://github.com/junghyunl.png" width="100"> | <img src="https://github.com/bin5459.png" width="100"> |
|                         **AI**                          |                          **BE**                           |                          **BE**                           |                          **AI**                           |                         **FE**                            |                           **FE**                           |
|        [@seul1230](https://github.com/seul1230)         |       [@BrokenFinger98](https://github.com/BrokenFinger98)        |       [@swoolee97](https://github.com/swoolee97)        |        [@LEE-SIU](https://github.com/LEE-SIU)       |       [@junghyunl](https://github.com/junghyunl)        |       [@bin5459](https://github.com/bin5459)       |

<br>

## 📌 브랜치
- 🚀 AI-dev : 엄마 AI, 보이스피싱(음성/내용 기반), 악성 URL 분류 모델 개발 및 데이터 전처리를 담당하는 브랜치입니다.
- 📱 APP-dev : 통화 중 녹음 감지, 문자 내용 중 하이퍼링크 감지 -> model inference (경량화된 AI 모델 탑재)
- ⭐ BE-dev
- 🎨 FE-dev



