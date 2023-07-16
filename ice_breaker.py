from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from third_party.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

information = """
박지성(한국 한자: 朴智星, 1981년 3월 30일 또는 1981년 음력 2월 25일)[2]은 대한민국의 은퇴한 프로 축구 선수로 현역 시절 포지션은 윙어, 미드필더였다. 현재 전북 현대 모터스의 테크니컬 디렉터 겸 퀸스 파크 레인저스 U-16 코치로 재직 중이다. 서울에서 태어난 그는 선수로 활동하는 동안 트로피 19개를 획득했다. UEFA 챔피언스리그에서 우승하고 챔피언스리그 결승전에 진출한 최초의 아시아 축구 선수이자 FIFA 클럽 월드컵에서 우승을 한 최초의 아시아인 선수이다. 그는 뛰어난 체력과 훈련, 프로 의식으로 유명했으며, 그의 지구력으로 인해 "폐 3개 가진 박(Three-Lungs Park)"이라는 별명을 얻었다.

어린 시절부터 축구를 시작한 그는 명지대학교 축구부에서 활동했으며, 2000년에 일본으로 건너가 교토 퍼플 상가에서 활동하며 프로 선수 경력을 시작했다. 그 후 2003년에 대한민국 국가대표팀 감독을 맡았던 거스 히딩크가 네덜란드로 돌아와 감독을 맡은 팀인 네덜란드의 PSV 에인트호번에 입단하여 유럽 리그로 진출을 했다. PSV가 2004-05년 UEFA 챔피언스리그 준결승에 진출한 후 맨체스터 유나이티드 FC의 감독 알렉스 퍼거슨에게 인정을 받아 2005년 7월에 맨체스터 유나이티드와 계약을 맺었다. 그는 프리미어리그에서 4번 우승했으며, 2007-08년 UEFA 챔피언스리그, 2008년 FIFA 클럽 월드컵에서 우승하는 데에 기여하였다. 이후 주전 출전 횟수가 감소하자 2012년 7월 퀸스 파크 레인저스 FC로 이적했다. 그러나 이적 시즌에 자신의 부상과 소속팀의 강등으로 인해 2013-14 시즌에 임대 형식으로 PSV 에인트호번에 합류했다. 이후 2014년에 PSV에서의 활동을 마지막으로 현역에서 은퇴했다.

대한민국 국가대표팀의 일원으로도 활동하여 A매치 100경기에 출전하여 13골을 넣었다. 그는 2002년 FIFA 월드컵에서 4위를 한 대한민국팀의 일원이었으며, 2006년 FIFA 월드컵과 2010년 FIFA 월드컵에서도 대한민국 국가대표로 참가하였다. 그는 월드컵에서 맨 오브더 매치(팬투표)에 3회 선정되었으며, 손흥민, 안정환과 함께 3골(14경기)로 한국 선수 월드컵 최다 득점자이다. 현재는 국제축구평의회 자문위원, 전북 현대 모터스 테크니컬 디렉터로 활동하고 있다.
"""

if __name__ == "__main__":
    print("Hello, Langchain")

    linkedin_profile_url = linkedin_lookup_agent(name="조코딩")
    print(linkedin_profile_url)

    summary_template = """
        주어지는 정보 {information} 는 내가 알고자 하는 사람의 정보 입니다
        1. 인물의 요약된 정보
        2. 두가지 인물에 대한 흥미로운 정보

        주어진 정보의 인물을 요약해줘
        대신 출력을 적절하게 보고서 형태로 해줘
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    # linkedin_data = scrape_linkedin_profile(
    #     linkedin_profile_url="https://www.linkedin.com/in/jocoding/"
    # )

    print(chain.run(information=information))
