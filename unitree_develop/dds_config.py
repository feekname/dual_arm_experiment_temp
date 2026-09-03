<CycloneDDS xmlns="https://cyclonedds.org/schema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="https://cyclonedds.org/schema https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd">
    <Domain id="any">
        <General>
            <Interfaces>
                <NetworkInterface name="enp89s0"/>
            </Interfaces>
            <AllowMulticast>true</AllowMulticast>
            <MaxMessageSize>65500B</MaxMessageSize>
        </General>
        <Discovery>
            <Peers>
                <Peer address="192.168.123.164"/>
            </Peers>
            <ParticipantIndex>auto</ParticipantIndex>
        </Discovery>
    </Domain>
</CycloneDDS>